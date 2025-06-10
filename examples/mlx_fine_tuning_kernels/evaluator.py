"""
MLX LoRA Fine-tuning Optimization Evaluator with Artifacts Support

This evaluator performs real LoRA fine-tuning benchmarks using the mlx-lm library,
comparing standard MLX-LM against MLX-LM with evolved kernels injected.
The goal is to achieve the same training loss with improved memory efficiency and/or speed.

Enhanced with artifacts to provide execution output feedback during evolution.
"""

import importlib.util
import time
import traceback
import statistics
import gc
import psutil
import os
import tempfile
import shutil
import json
import sys
import io
import contextlib
from typing import Dict, Union, List, Tuple, Optional, Any
from pathlib import Path

# Import EvaluationResult for artifacts support
from openevolve.evaluation_result import EvaluationResult

# Required imports - fail fast if not available
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
except ImportError as e:
    raise ImportError(f"MLX not available: {e}. Please install with: pip install mlx")

try:
    import psutil
except ImportError as e:
    raise ImportError(f"psutil not available: {e}. Please install with: pip install psutil")

try:
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        print_trainable_parameters,
    )
    from mlx_lm.utils import save_config

    MLX_LM_AVAILABLE = True
    print("✅ MLX-LM available for evaluation")
except ImportError as e:
    print(f"⚠️ MLX-LM not available: {e}")
    MLX_LM_AVAILABLE = False


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def clear_mlx_cache_and_gc():
    """Clear MLX cache and run garbage collection."""
    mx.clear_cache()
    gc.collect()


@contextlib.contextmanager
def capture_output():
    """Context manager to capture stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class MLXLoRABenchmark:
    """
    Benchmark for comparing standard MLX-LM vs MLX-LM with evolved kernels.
    Uses proper sequential evaluation to avoid monkey patching interference.
    """

    def __init__(self, model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        self.model_name = model_name
        self.temp_dirs = []

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        self.temp_dirs.clear()

    def create_test_config(self, data_dir: str, adapter_dir: str) -> Dict[str, Any]:
        """Create test configuration for LoRA fine-tuning with all MLX-LM expected attributes."""
        return {
            "model": self.model_name,
            "train": True,
            "fine_tune_type": "lora",
            "optimizer": "adam",
            "optimizer_config": {"adam": {}},
            "data": data_dir,
            "seed": 42,
            "num_layers": 4,  # More layers for comprehensive evaluation
            "batch_size": 2,  # Reasonable batch size for larger dataset
            "iters": 25,  # More iterations for larger dataset
            "val_batches": 10,
            "learning_rate": 1e-4,
            "steps_per_report": 10,
            "steps_per_eval": 50,
            "adapter_path": adapter_dir,
            "save_every": 100,
            "max_seq_length": 512,  # Full sequence length
            "lora_parameters": {"rank": 16, "dropout": 0.0, "scale": 16.0},  # Standard rank
            "mask_prompt": False,
            # Additional MLX-LM expected attributes
            "test": True,
            "test_batches": 10,
            "resume_adapter_file": None,
            "config": None,
            "grad_checkpoint": False,
            "lr_schedule": None,
            "wandb": None,
        }

    def compare_implementations(self, evolved_kernels: Dict, num_trials: int = 3) -> Dict[str, Any]:
        """
        Compare standard MLX-LM vs MLX-LM with evolved kernels.

        PROPER EVALUATION STRUCTURE:
        1. Run ALL baseline trials first (no patching)
        2. Calculate baseline metrics
        3. Apply evolved kernels patching ONCE
        4. Run ALL evolved trials
        5. Calculate evolved metrics
        6. Compare results

        This avoids monkey patching interference between trials.
        """

        if not MLX_LM_AVAILABLE:
            return {"error": "MLX-LM not available for real benchmarking"}

        print(f"\n📊 MLX-LM LORA KERNEL COMPARISON")
        print(f"  Model: {self.model_name}")
        print(f"  Trials per implementation: {num_trials}")
        print(f"  Evaluation strategy: Sequential (baseline first, then evolved)")
        print(
            f"  Evolved kernels available: {list(evolved_kernels.keys()) if evolved_kernels else 'None'}"
        )

        baseline_results = []
        evolved_results = []

        # ========================================
        # PHASE 1: Run ALL baseline trials first
        # ========================================
        print(f"\n🔬 PHASE 1: Running {num_trials} BASELINE trials (standard MLX-LM)")

        for trial in range(num_trials):
            print(f"\n--- Baseline Trial {trial + 1}/{num_trials} ---")

            # Create temporary directories for this trial
            baseline_data_dir = tempfile.mkdtemp(prefix="baseline_data_")
            baseline_adapter_dir = tempfile.mkdtemp(prefix="baseline_adapters_")
            self.temp_dirs.extend([baseline_data_dir, baseline_adapter_dir])

            try:
                # Create test dataset
                self._create_test_dataset(baseline_data_dir)
                baseline_config = self.create_test_config(baseline_data_dir, baseline_adapter_dir)

                clear_mlx_cache_and_gc()

                # Run baseline (standard MLX-LM)
                baseline_result = self._run_single_trial(
                    baseline_config,
                    f"BASELINE-{trial+1}",
                    evolved_kernels=None,  # No kernels = standard MLX-LM
                )
                baseline_results.append(baseline_result)

                # Early exit if first baseline trial fails
                if trial == 0 and "error" in baseline_result:
                    print("  🚨 First baseline trial failed - stopping evaluation")
                    return {"error": f"First baseline trial failed: {baseline_result['error']}"}

            except Exception as e:
                print(f"  ❌ Baseline trial {trial+1} failed: {e}")
                baseline_results.append({"error": str(e)})

                # Early exit if first trial fails
                if trial == 0:
                    print("  🚨 First baseline trial failed - stopping evaluation")
                    return {"error": f"First baseline trial failed: {e}"}

        # ========================================
        # PHASE 2: Run ALL evolved trials
        # ========================================
        print(f"\n🚀 PHASE 2: Running {num_trials} EVOLVED trials (MLX-LM + evolved kernels)")

        # Verify evolved kernels are valid before running trials
        if evolved_kernels:
            print(f"  ✅ Testing evolved kernels: {list(evolved_kernels.keys())}")
            for kernel_name, kernel_func in evolved_kernels.items():
                if kernel_func is None:
                    print(f"  ⚠️ Warning: {kernel_name} is None")
                else:
                    print(f"  ✅ {kernel_name}: {type(kernel_func)}")

        for trial in range(num_trials):
            print(f"\n--- Evolved Trial {trial + 1}/{num_trials} ---")

            # Create temporary directories for this trial
            evolved_data_dir = tempfile.mkdtemp(prefix="evolved_data_")
            evolved_adapter_dir = tempfile.mkdtemp(prefix="evolved_adapters_")
            self.temp_dirs.extend([evolved_data_dir, evolved_adapter_dir])

            try:
                # Create test dataset (same as baseline)
                self._create_test_dataset(evolved_data_dir)
                evolved_config = self.create_test_config(evolved_data_dir, evolved_adapter_dir)

                clear_mlx_cache_and_gc()

                # Run evolved (MLX-LM + evolved kernels)
                evolved_result = self._run_single_trial(
                    evolved_config,
                    f"EVOLVED-{trial+1}",
                    evolved_kernels=evolved_kernels,  # Inject evolved kernels
                )
                evolved_results.append(evolved_result)

                # Early exit if first evolved trial fails
                if trial == 0 and "error" in evolved_result:
                    print("  🚨 First evolved trial failed - stopping evaluation")
                    return {"error": f"First evolved trial failed: {evolved_result['error']}"}

            except Exception as e:
                print(f"  ❌ Evolved trial {trial+1} failed: {e}")
                evolved_results.append({"error": str(e)})

                # Early exit if first trial fails
                if trial == 0:
                    print("  🚨 First evolved trial failed - stopping evaluation")
                    return {"error": f"First evolved trial failed: {e}"}

        # ========================================
        # PHASE 3: Analyze and compare results
        # ========================================
        self.cleanup()

        results = {"baseline": baseline_results, "evolved": evolved_results}

        return self._analyze_results(results)

    def _create_test_dataset(self, output_dir: str, num_samples: int = 500):
        """Create a comprehensive test dataset for LoRA fine-tuning with diverse examples."""
        examples = [
            # AI and Machine Learning
            {
                "text": "What is AI?\nAI is artificial intelligence, a field where computers perform tasks that typically require human intelligence."
            },
            {
                "text": "How does ML work?\nMachine learning involves algorithms learning patterns from data to make predictions or decisions."
            },
            {
                "text": "What is Python?\nPython is a versatile, high-level programming language known for its readability and simplicity."
            },
            {
                "text": "Explain deep learning.\nDeep learning uses neural networks with multiple layers to model complex patterns in data."
            },
            {
                "text": "What is NLP?\nNatural Language Processing enables computers to understand and generate human language."
            },
            {
                "text": "What is a neural network?\nA neural network is a computing system inspired by biological neural networks that learns from data."
            },
            {
                "text": "What is supervised learning?\nSupervised learning trains models on labeled data to predict outcomes for new data."
            },
            {
                "text": "What is unsupervised learning?\nUnsupervised learning finds patterns in unlabeled data without predefined outcomes."
            },
            {
                "text": "What is reinforcement learning?\nReinforcement learning trains agents to make decisions by rewarding desired behaviors."
            },
            {
                "text": "What is a transformer model?\nA transformer model processes sequential data using attention mechanisms, common in NLP."
            },
            {
                "text": "What is computer vision?\nComputer vision enables computers to interpret and understand visual information from images and videos."
            },
            {
                "text": "What is data science?\nData science extracts insights from data using statistics, programming, and domain expertise."
            },
            {
                "text": "What is a decision tree?\nA decision tree is a model that makes decisions by splitting data based on feature values."
            },
            {
                "text": "What is overfitting?\nOverfitting occurs when a model learns training data too well, reducing its ability to generalize."
            },
            {
                "text": "What is cross-validation?\nCross-validation assesses model performance by splitting data into training and testing sets."
            },
            # Programming and Technology
            {
                "text": "What is a database?\nA database is an organized collection of data, typically stored and accessed electronically."
            },
            {
                "text": "What is cloud computing?\nCloud computing delivers computing services over the internet, providing scalability and flexibility."
            },
            {
                "text": "What is blockchain?\nBlockchain is a decentralized ledger technology that ensures secure and transparent transactions."
            },
            {
                "text": "What is an API?\nAn API is an interface that allows different software applications to communicate with each other."
            },
            {
                "text": "What is a GPU?\nA Graphics Processing Unit is specialized hardware for accelerating computations, often used in AI."
            },
            {
                "text": "What is quantum computing?\nQuantum computing uses quantum mechanics to perform computations, potentially solving problems faster than classical computers."
            },
            {
                "text": "What is cybersecurity?\nCybersecurity protects computer systems, networks, and data from digital attacks and unauthorized access."
            },
            {
                "text": "What is DevOps?\nDevOps combines software development and IT operations to improve collaboration and deployment efficiency."
            },
            {
                "text": "What is version control?\nVersion control tracks changes to files over time, allowing multiple people to collaborate on projects."
            },
            {
                "text": "What is open source software?\nOpen source software has publicly available source code that anyone can view, modify, and distribute."
            },
            {
                "text": "What is a web browser?\nA web browser is software that allows users to access and navigate websites on the internet."
            },
            {
                "text": "What is JavaScript?\nJavaScript is a programming language commonly used for web development and interactive websites."
            },
            {
                "text": "What is mobile app development?\nMobile app development creates software applications designed to run on smartphones and tablets."
            },
            {
                "text": "What is artificial neural networks?\nArtificial neural networks are computing systems inspired by biological neural networks in animal brains."
            },
            {
                "text": "What is the Internet of Things?\nThe Internet of Things connects everyday devices to the internet, enabling data collection and automation."
            },
            # Science and Nature
            {
                "text": "What is photosynthesis?\nPhotosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
            },
            {
                "text": "What is DNA?\nDNA is the molecule that carries genetic instructions for the development and functioning of living organisms."
            },
            {
                "text": "What is climate change?\nClimate change refers to long-term shifts in global temperatures and weather patterns due to human activities."
            },
            {
                "text": "What is renewable energy?\nRenewable energy comes from natural sources that replenish themselves, like solar, wind, and hydroelectric power."
            },
            {
                "text": "What is evolution?\nEvolution is the process by which species change over time through natural selection and genetic variation."
            },
            {
                "text": "What is the periodic table?\nThe periodic table organizes chemical elements by their atomic number and properties in a systematic arrangement."
            },
            {
                "text": "What is gravity?\nGravity is a fundamental force that attracts objects with mass toward each other, keeping us on Earth."
            },
            {
                "text": "What is the water cycle?\nThe water cycle describes how water moves through Earth's systems via evaporation, condensation, and precipitation."
            },
            {
                "text": "What is biodiversity?\nBiodiversity refers to the variety of life forms in an ecosystem, including species, genetic, and ecosystem diversity."
            },
            {
                "text": "What is an ecosystem?\nAn ecosystem is a community of living organisms interacting with their physical environment."
            },
            {
                "text": "What is conservation?\nConservation involves protecting and preserving natural resources and wildlife for future generations."
            },
            {
                "text": "What is astronomy?\nAstronomy is the scientific study of celestial objects, space, and the universe as a whole."
            },
            {
                "text": "What is geology?\nGeology studies the Earth's physical structure, substances, history, and the processes that act on them."
            },
            {
                "text": "What is marine biology?\nMarine biology studies organisms in the ocean and other saltwater environments."
            },
            {
                "text": "What is meteorology?\nMeteorology is the study of weather patterns, atmospheric conditions, and climate systems."
            },
            # Health and Medicine
            {
                "text": "What is the immune system?\nThe immune system defends the body against infections and diseases through specialized cells and organs."
            },
            {
                "text": "What are vitamins?\nVitamins are essential nutrients that the body needs in small amounts for proper growth and function."
            },
            {
                "text": "What is exercise?\nExercise is physical activity that improves fitness, health, and overall well-being."
            },
            {
                "text": "What is nutrition?\nNutrition is the process of obtaining and consuming food necessary for health and growth."
            },
            {
                "text": "What is mental health?\nMental health encompasses emotional, psychological, and social well-being affecting how we think and feel."
            },
            {
                "text": "What is meditation?\nMeditation is a practice that focuses the mind to achieve mental clarity, emotional stability, and relaxation."
            },
            {
                "text": "What are antibiotics?\nAntibiotics are medicines that fight bacterial infections by killing bacteria or stopping their growth."
            },
            {
                "text": "What is vaccination?\nVaccination introduces weakened or inactive parts of organisms to stimulate immune system protection against diseases."
            },
            {
                "text": "What is stress?\nStress is the body's response to challenging or demanding situations, affecting both physical and mental health."
            },
            {
                "text": "What is sleep?\nSleep is a natural state of rest that allows the body and mind to recover and maintain essential functions."
            },
            {
                "text": "What is diabetes?\nDiabetes is a condition where the body cannot properly process blood glucose due to insulin problems."
            },
            {
                "text": "What is cardiovascular health?\nCardiovascular health refers to the well-being of the heart and blood vessels in the circulatory system."
            },
            {
                "text": "What is physical therapy?\nPhysical therapy helps restore movement and function when someone is affected by injury, illness, or disability."
            },
            {
                "text": "What is public health?\nPublic health focuses on protecting and improving the health of entire populations and communities."
            },
            {
                "text": "What is preventive medicine?\nPreventive medicine focuses on preventing diseases and health problems before they occur."
            },
            # Geography and Culture
            {"text": "What is the capital of France?\nThe capital of France is Paris."},
            {
                "text": "What is the Great Wall of China?\nThe Great Wall of China is an ancient series of walls and fortifications built to protect Chinese states."
            },
            {
                "text": "What is democracy?\nDemocracy is a system of government where citizens exercise power through voting and elected representatives."
            },
            {
                "text": "What is globalization?\nGlobalization is the increasing interconnectedness of countries through trade, culture, and communication."
            },
            {
                "text": "What is culture?\nCulture encompasses the beliefs, customs, arts, and social behaviors of a particular group or society."
            },
            {
                "text": "What is the United Nations?\nThe United Nations is an international organization that promotes peace, security, and cooperation among nations."
            },
            {
                "text": "What is the European Union?\nThe European Union is a political and economic union of European countries promoting integration and cooperation."
            },
            {
                "text": "What is the Amazon rainforest?\nThe Amazon rainforest is the world's largest tropical rainforest, playing a crucial role in global climate regulation."
            },
            {
                "text": "What is the Pacific Ocean?\nThe Pacific Ocean is the largest and deepest ocean on Earth, covering about one-third of the planet's surface."
            },
            {
                "text": "What is Mount Everest?\nMount Everest is the highest mountain peak on Earth, located in the Himalayas between Nepal and Tibet."
            },
            {
                "text": "What is urbanization?\nUrbanization is the process of population shift from rural to urban areas, leading to city growth."
            },
            {
                "text": "What is migration?\nMigration is the movement of people from one place to another, often for economic or social reasons."
            },
            {
                "text": "What is archaeology?\nArchaeology studies human history through the excavation and analysis of artifacts and other physical remains."
            },
            {
                "text": "What is anthropology?\nAnthropology is the study of human societies, cultures, and their development over time."
            },
            {
                "text": "What is linguistics?\nLinguistics is the scientific study of language and its structure, evolution, and use."
            },
            # Mathematics and Physics
            {
                "text": "What is algebra?\nAlgebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in equations."
            },
            {
                "text": "What is geometry?\nGeometry is the branch of mathematics that deals with shapes, sizes, positions, and properties of space."
            },
            {
                "text": "What is calculus?\nCalculus is the mathematical study of continuous change, involving derivatives and integrals."
            },
            {
                "text": "What is statistics?\nStatistics is the science of collecting, analyzing, interpreting, and presenting data to make informed decisions."
            },
            {
                "text": "What is physics?\nPhysics is the science that studies matter, energy, motion, and the fundamental forces of the universe."
            },
            {
                "text": "What is electricity?\nElectricity is the flow of electric charge through conductors, powering countless devices and systems."
            },
            {
                "text": "What is magnetism?\nMagnetism is a physical phenomenon where certain materials attract or repel each other through magnetic fields."
            },
            {
                "text": "What is energy?\nEnergy is the capacity to do work or cause change, existing in many forms like kinetic, potential, and thermal."
            },
            {
                "text": "What is the speed of light?\nThe speed of light is approximately 299,792,458 meters per second in a vacuum, the fastest possible speed."
            },
            {
                "text": "What is relativity?\nRelativity is Einstein's theory describing how space and time are linked and affected by gravity and motion."
            },
            {
                "text": "What is thermodynamics?\nThermodynamics studies the relationships between heat, work, temperature, and energy in physical systems."
            },
            {
                "text": "What is quantum mechanics?\nQuantum mechanics describes the behavior of matter and energy at the atomic and subatomic scale."
            },
            {
                "text": "What is probability?\nProbability measures the likelihood of events occurring, expressed as numbers between 0 and 1."
            },
            {
                "text": "What is trigonometry?\nTrigonometry studies relationships between angles and sides of triangles, used in many applications."
            },
            {
                "text": "What is number theory?\nNumber theory is a branch of mathematics devoted to the study of integers and integer-valued functions."
            },
            # Business and Economics
            {
                "text": "What is entrepreneurship?\nEntrepreneurship is the process of creating and managing a business venture to generate profit and innovation."
            },
            {
                "text": "What is marketing?\nMarketing involves promoting and selling products or services by understanding and meeting customer needs."
            },
            {
                "text": "What is economics?\nEconomics studies how societies allocate scarce resources to satisfy unlimited wants and needs."
            },
            {
                "text": "What is inflation?\nInflation is the general increase in prices of goods and services over time, reducing purchasing power."
            },
            {
                "text": "What is supply and demand?\nSupply and demand are economic forces that determine the price and quantity of goods in a market."
            },
            {
                "text": "What is cryptocurrency?\nCryptocurrency is digital money secured by cryptography and typically based on blockchain technology."
            },
            {
                "text": "What is e-commerce?\nE-commerce is the buying and selling of goods and services over the internet through digital platforms."
            },
            {
                "text": "What is leadership?\nLeadership is the ability to guide, motivate, and influence others toward achieving common goals."
            },
            {
                "text": "What is teamwork?\nTeamwork is the collaborative effort of individuals working together to accomplish shared objectives."
            },
            {
                "text": "What is innovation?\nInnovation is the process of creating new ideas, products, or methods that provide value and solve problems."
            },
            {
                "text": "What is investment?\nInvestment involves allocating money or resources with the expectation of generating income or profit."
            },
            {
                "text": "What is financial planning?\nFinancial planning involves managing money and assets to achieve personal financial goals and security."
            },
            {
                "text": "What is project management?\nProject management coordinates resources, tasks, and timelines to achieve specific objectives within constraints."
            },
            {
                "text": "What is human resources?\nHuman resources manages employee relations, recruitment, training, and organizational development."
            },
            {
                "text": "What is strategic planning?\nStrategic planning defines long-term goals and determines the best approach to achieve them."
            },
            # Arts and Literature
            {
                "text": "What is art?\nArt is the expression of human creativity and imagination through various mediums like painting, sculpture, and music."
            },
            {
                "text": "What is literature?\nLiterature comprises written works of artistic merit, including novels, poetry, and plays that express human experience."
            },
            {
                "text": "What is music?\nMusic is the art of organizing sounds in time through rhythm, melody, harmony, and expression."
            },
            {
                "text": "What is photography?\nPhotography is the art and science of capturing light to create images that document or express visual ideas."
            },
            {
                "text": "What is theater?\nTheater is the performance of stories through acting, dialogue, music, and stagecraft for live audiences."
            },
            {
                "text": "What is poetry?\nPoetry is literary art that uses aesthetic and rhythmic language to express emotions, ideas, and experiences."
            },
            {
                "text": "What is architecture?\nArchitecture is the art and science of designing and constructing buildings and other physical structures."
            },
            {
                "text": "What is sculpture?\nSculpture is the art of creating three-dimensional works by carving, modeling, or assembling materials."
            },
            {
                "text": "What is dance?\nDance is the art of movement through space and time, often accompanied by music and expressing emotions."
            },
            {
                "text": "What is film?\nFilm is the art of creating moving pictures that tell stories through visual and auditory elements."
            },
            {
                "text": "What is creative writing?\nCreative writing is the art of crafting original works that express ideas, emotions, and stories imaginatively."
            },
            {
                "text": "What is graphic design?\nGraphic design combines text, images, and visual elements to communicate messages effectively."
            },
            {
                "text": "What is interior design?\nInterior design plans and designs interior spaces to be functional, safe, and aesthetically pleasing."
            },
            {
                "text": "What is fashion design?\nFashion design creates clothing and accessories that combine function, style, and artistic expression."
            },
            {
                "text": "What is digital art?\nDigital art uses digital technology as an essential part of the creative or presentation process."
            },
            # History and Philosophy
            {
                "text": "What is history?\nHistory is the study of past events, their causes, and their impact on human civilization."
            },
            {
                "text": "What is philosophy?\nPhilosophy is the study of fundamental questions about existence, knowledge, values, and human nature."
            },
            {
                "text": "What is the Renaissance?\nThe Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries, marked by art and learning."
            },
            {
                "text": "What is the Industrial Revolution?\nThe Industrial Revolution was a period of major industrialization and innovation that transformed society from agriculture to manufacturing."
            },
            {
                "text": "What is democracy in ancient Greece?\nAncient Greek democracy was a system where citizens participated directly in political decision-making in city-states like Athens."
            },
            {
                "text": "What is ethics?\nEthics is the branch of philosophy that deals with moral principles and determining right and wrong behavior."
            },
            {
                "text": "What is logic?\nLogic is the systematic study of the principles of valid reasoning and correct inference."
            },
            {
                "text": "What is existentialism?\nExistentialism is a philosophical movement emphasizing individual existence, freedom, and the meaning of life."
            },
            {
                "text": "What is the Enlightenment?\nThe Enlightenment was an 18th-century intellectual movement emphasizing reason, science, and individual rights."
            },
            {
                "text": "What is the Scientific Revolution?\nThe Scientific Revolution was a period of major advances in scientific thought and methodology in the 16th and 17th centuries."
            },
            {
                "text": "What is world history?\nWorld history studies the development of human civilization across all regions and time periods globally."
            },
            {
                "text": "What is political science?\nPolitical science examines government systems, political behavior, and the theory and practice of politics."
            },
            {
                "text": "What is sociology?\nSociology studies human society, social relationships, and the forces that shape social behavior."
            },
            {
                "text": "What is psychology?\nPsychology is the scientific study of mind and behavior, including cognitive, emotional, and social processes."
            },
            {
                "text": "What is theology?\nTheology is the study of religious beliefs, practices, and the nature of the divine."
            },
            # Food and Cooking
            {
                "text": "How do you make tea?\nTo make tea, boil water, add tea leaves or a tea bag to a cup, pour the hot water over the tea, let it steep for 3-5 minutes, then remove the tea leaves or bag."
            },
            {
                "text": "How do you cook pasta?\nTo cook pasta, boil salted water, add pasta and cook according to package directions, then drain and serve with sauce."
            },
            {
                "text": "What is nutrition science?\nNutrition science studies how food affects the body, providing essential nutrients for growth, energy, and health."
            },
            {
                "text": "What is organic food?\nOrganic food is produced without synthetic pesticides, fertilizers, or genetic modification, following natural farming practices."
            },
            {
                "text": "What is vegetarianism?\nVegetarianism is a diet that excludes meat, focusing on plant-based foods for health, ethical, or environmental reasons."
            },
            {
                "text": "What is fermentation?\nFermentation is a process where microorganisms convert sugars into acids, gases, or alcohol, used in food preservation."
            },
            {
                "text": "What is baking?\nBaking is cooking food using dry heat in an oven, commonly used for bread, cakes, and pastries."
            },
            {
                "text": "What are spices?\nSpices are aromatic plant substances used to flavor, color, and preserve food, derived from seeds, bark, or roots."
            },
            {
                "text": "What is sustainable farming?\nSustainable farming practices maintain soil health and environmental balance while producing food efficiently."
            },
            {
                "text": "What is food safety?\nFood safety involves proper handling, preparation, and storage of food to prevent contamination and foodborne illness."
            },
            {
                "text": "What is culinary arts?\nCulinary arts involve the preparation, cooking, and presentation of food as both sustenance and artistic expression."
            },
            {
                "text": "What is agriculture?\nAgriculture is the cultivation of plants and livestock for food, fiber, and other products used to sustain life."
            },
            {
                "text": "What is gastronomy?\nGastronomy is the art and science of good eating, including the study of food and culture relationships."
            },
            {
                "text": "What is food chemistry?\nFood chemistry studies the chemical processes and interactions of biological and non-biological components in food."
            },
            {
                "text": "What is dietetics?\nDietetics applies nutrition science to promote health and treat disease through proper food and eating habits."
            },
        ]

        # Use smaller dataset for faster evaluation
        if num_samples > len(examples):
            dataset = []
            for i in range(num_samples):
                dataset.append(examples[i % len(examples)])
        else:
            dataset = examples[:num_samples]

        # Create balanced splits with minimum sizes
        train_size = max(10, int(0.7 * num_samples))
        val_size = max(5, int(0.2 * num_samples))
        test_size = max(3, num_samples - train_size - val_size)

        train_data = dataset[:train_size]
        val_data = dataset[train_size : train_size + val_size]
        test_data = dataset[train_size + val_size : train_size + val_size + test_size]

        print(
            f"📊 Dataset: {len(train_data)} train, {len(val_data)} valid, {len(test_data)} test examples"
        )

        # Write datasets - Use "valid" not "val" for MLX-LM
        os.makedirs(output_dir, exist_ok=True)
        for split, data in [("train", train_data), ("valid", val_data), ("test", test_data)]:
            file_path = os.path.join(output_dir, f"{split}.jsonl")
            with open(file_path, "w") as f:
                for example in data:
                    f.write(json.dumps(example) + "\n")

    def _run_single_trial(
        self, config: Dict[str, Any], trial_name: str, evolved_kernels: Optional[Dict] = None
    ) -> Dict[str, Union[float, str]]:
        """Run a single LoRA fine-tuning trial."""

        print(f"  🧪 Running {trial_name}...")
        if evolved_kernels:
            print(f"    📦 Using evolved kernels: {list(evolved_kernels.keys())}")
        else:
            print(f"    📋 Using standard MLX-LM (no kernels)")

        try:
            # Memory before
            memory_before = get_memory_usage()
            start_time = time.perf_counter()

            # Import and run the training function
            import sys
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)

            from initial_program import standard_lora_fine_tuning_with_kernels

            # Run training with or without evolved kernels
            final_loss, metrics = standard_lora_fine_tuning_with_kernels(
                model_name=config["model"],
                train_data_path=config["data"],
                config=config,
                adapter_save_path=config["adapter_path"],
                evolved_kernels=evolved_kernels,
            )

            # Timing and memory
            end_time = time.perf_counter()
            memory_after = get_memory_usage()

            total_time = end_time - start_time
            memory_delta = memory_after - memory_before

            # Extract additional metrics
            training_time = metrics.get("training_time", total_time)

            # Check if kernels were actually used
            kernels_used = metrics.get("used_evolved_kernels", False)
            if evolved_kernels and not kernels_used:
                print(f"    ⚠️ Warning: Evolved kernels provided but not used")
            elif evolved_kernels and kernels_used:
                print(f"    ✅ Evolved kernels successfully applied")

            # Calculate approximate tokens/second
            estimated_tokens = config["iters"] * config["batch_size"] * config["max_seq_length"]
            tokens_per_second = estimated_tokens / training_time if training_time > 0 else 0

            print(f"    Final loss: {final_loss:.4f}")
            print(f"    Training time: {training_time:.2f}s")
            print(f"    Memory delta: {memory_delta:.1f} MB")
            print(f"    Tokens/sec: {tokens_per_second:.1f}")
            print(f"    Kernels used: {kernels_used}")

            return {
                "final_loss": float(final_loss),
                "training_time": float(training_time),
                "total_time": float(total_time),
                "memory_delta": float(memory_delta),
                "tokens_per_second": float(tokens_per_second),
                "lora_rank": config["lora_parameters"]["rank"],
                "num_layers": config["num_layers"],
                "kernels_used": bool(kernels_used),
            }

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    def _analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze comparison results."""

        # Filter successful results
        baseline_success = [r for r in results["baseline"] if "error" not in r]
        evolved_success = [r for r in results["evolved"] if "error" not in r]

        if not baseline_success or not evolved_success:
            return {
                "error": "No successful trials for comparison",
                "baseline_success": len(baseline_success),
                "evolved_success": len(evolved_success),
            }

        # Calculate averages
        baseline_avg = {
            "final_loss": np.mean([r["final_loss"] for r in baseline_success]),
            "training_time": np.mean([r["training_time"] for r in baseline_success]),
            "memory_delta": np.mean([r["memory_delta"] for r in baseline_success]),
            "tokens_per_second": np.mean([r["tokens_per_second"] for r in baseline_success]),
        }

        evolved_avg = {
            "final_loss": np.mean([r["final_loss"] for r in evolved_success]),
            "training_time": np.mean([r["training_time"] for r in evolved_success]),
            "memory_delta": np.mean([r["memory_delta"] for r in evolved_success]),
            "tokens_per_second": np.mean([r["tokens_per_second"] for r in evolved_success]),
        }

        # Calculate improvements
        loss_difference = abs(evolved_avg["final_loss"] - baseline_avg["final_loss"])
        loss_tolerance = max(0.01 * baseline_avg["final_loss"], 0.001)  # 1% or 0.001 minimum
        loss_convergence_ok = loss_difference <= loss_tolerance

        speed_improvement = (
            evolved_avg["tokens_per_second"] / baseline_avg["tokens_per_second"]
            if baseline_avg["tokens_per_second"] > 0
            else 1.0
        )
        time_improvement = (
            baseline_avg["training_time"] / evolved_avg["training_time"]
            if evolved_avg["training_time"] > 0
            else 1.0
        )
        memory_improvement = (
            baseline_avg["memory_delta"] / evolved_avg["memory_delta"]
            if evolved_avg["memory_delta"] > 0
            else 1.0
        )

        # Overall score calculation
        convergence_score = (
            1.0
            if loss_convergence_ok
            else max(0.0, 1.0 - (loss_difference / baseline_avg["final_loss"]))
        )
        efficiency_score = 0.5 * min(speed_improvement / 1.05, 2.0) + 0.5 * min(
            memory_improvement / 1.05, 2.0
        )
        overall_score = 0.7 * convergence_score + 0.3 * efficiency_score

        # Check if kernels were actually used in evolved trials
        kernels_actually_used = any(r.get("kernels_used", False) for r in evolved_success)

        return {
            "baseline_avg": baseline_avg,
            "evolved_avg": evolved_avg,
            "loss_difference": loss_difference,
            "loss_convergence_ok": loss_convergence_ok,
            "speed_improvement": speed_improvement,
            "time_improvement": time_improvement,
            "memory_improvement": memory_improvement,
            "convergence_score": convergence_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall_score,
            "successful_trials": {
                "baseline": len(baseline_success),
                "evolved": len(evolved_success),
            },
            "kernels_actually_used": kernels_actually_used,
            "evolved_trials_debug": evolved_success,
        }


def evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate MLX-LM LoRA kernel optimization program.

    Returns:
        EvaluationResult with metrics and artifacts (stdout/stderr) for evolution feedback
    """
    print(f"🚀 Evaluating MLX LoRA Kernel Optimization: {program_path}")

    if not MLX_LM_AVAILABLE:
        return EvaluationResult(
            metrics={"overall_score": 0.0},
            artifacts={
                "stderr": "MLX-LM not available for evaluation. Please install: pip install mlx-lm",
            }
        )

    # Capture all output during evaluation
    with capture_output() as (stdout_capture, stderr_capture):
        try:
            # Load evolved program
            spec = importlib.util.spec_from_file_location("evolved_program", program_path)
            evolved_program = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evolved_program)

            if not hasattr(evolved_program, "evolved_lora_kernels"):
                return EvaluationResult(
                    metrics={"overall_score": 0.0},
                    artifacts={
                        "stderr": "Missing evolved_lora_kernels function",
                    }
                )

            if not hasattr(evolved_program, "baseline_lora_kernels"):
                return EvaluationResult(
                    metrics={"overall_score": 0.0},
                    artifacts={
                        "stderr": "Missing baseline_lora_kernels function",
                    }
                )

            # Get evolved kernels
            print("📦 Loading evolved kernels...")
            try:
                evolved_kernels = evolved_program.evolved_lora_kernels()
                baseline_kernels = evolved_program.baseline_lora_kernels()  # Returns None

                print(
                    f"✅ Evolved kernels loaded: {list(evolved_kernels.keys()) if evolved_kernels else 'None'}"
                )
                print(f"✅ Baseline: Standard MLX-LM (no custom kernels)")

                # Validate evolved kernels
                if evolved_kernels:
                    for kernel_name, kernel_func in evolved_kernels.items():
                        if kernel_func is None:
                            print(f"  ⚠️ Warning: {kernel_name} is None")
                        else:
                            print(f"  ✅ {kernel_name}: {type(kernel_func)}")

            except Exception as e:
                print(f"❌ Failed to load evolved kernels: {e}")
                return EvaluationResult(
                    metrics={"overall_score": 0.0},
                    artifacts={
                        "stderr": f"Failed to load evolved kernels: {e}",
                        "traceback": traceback.format_exc(),
                    }
                )

            # Setup benchmark
            benchmark = MLXLoRABenchmark()

            # Run sequential comparison (baseline first, then evolved)
            comparison_results = benchmark.compare_implementations(
                evolved_kernels=evolved_kernels, num_trials=5
            )

            if "error" in comparison_results:
                return EvaluationResult(
                    metrics={"overall_score": 0.0},
                    artifacts={
                        "stderr": comparison_results["error"],
                    }
                )

            # Extract results
            overall_score = comparison_results["overall_score"]
            convergence_score = comparison_results["convergence_score"]
            efficiency_score = comparison_results["efficiency_score"]

            loss_difference = comparison_results["loss_difference"]
            loss_convergence_ok = comparison_results["loss_convergence_ok"]
            speed_improvement = comparison_results["speed_improvement"]
            memory_improvement = comparison_results["memory_improvement"]
            time_improvement = comparison_results["time_improvement"]

            baseline_avg = comparison_results["baseline_avg"]
            evolved_avg = comparison_results["evolved_avg"]

            print(f"\n📊 MLX LORA KERNEL OPTIMIZATION RESULTS:")
            print(
                f"  Loss Convergence: {'✅' if loss_convergence_ok else '❌'} (diff: {loss_difference:.4f})"
            )
            print(f"  Speed Improvement: {speed_improvement:.2f}x")
            print(f"  Memory Improvement: {memory_improvement:.2f}x")
            print(f"  Time Improvement: {time_improvement:.2f}x")
            print(f"  Convergence Score: {convergence_score:.3f}")
            print(f"  Efficiency Score: {efficiency_score:.3f}")
            print(f"  Overall Score: {overall_score:.3f}")

            print(f"\n🔍 DETAILED METRICS:")
            print(
                f"  Baseline - Loss: {baseline_avg['final_loss']:.4f}, Time: {baseline_avg['training_time']:.1f}s, Memory: {baseline_avg['memory_delta']:.1f} MB"
            )
            print(
                f"  Evolved  - Loss: {evolved_avg['final_loss']:.4f}, Time: {evolved_avg['training_time']:.1f}s, Memory: {evolved_avg['memory_delta']:.1f} MB"
            )

            # Check if kernels were actually used in evolved trials
            kernels_actually_used = comparison_results.get("kernels_actually_used", False)
            
            if evolved_kernels:
                if kernels_actually_used:
                    print(f"  ✅ Evolved kernels were successfully used in trials")
                else:
                    print(f"  ⚠️ WARNING: Evolved kernels were provided but not used in trials")

            # Success interpretation
            if overall_score >= 0.8:
                print("  🥇 EXCELLENT: Strong improvements while maintaining convergence!")
            elif overall_score >= 0.6:
                print("  🥈 VERY GOOD: Good improvements with convergence!")
            elif overall_score >= 0.4:
                print("  🥉 GOOD: Some improvements achieved!")
            elif convergence_score > 0.5:
                print("  📈 PROGRESS: Reasonable convergence, efficiency needs work!")
            else:
                print("  🔄 DEVELOPING: Convergence issues need to be addressed!")

            # Prepare metrics
            metrics = {
                "overall_score": float(overall_score),
                "combined_score": float(overall_score),  # Primary metric for OpenEvolve
                # Core metrics
                "convergence_score": float(convergence_score),
                "efficiency_score": float(efficiency_score),
                "loss_convergence_ok": bool(loss_convergence_ok),
                "loss_difference": float(loss_difference),
                # Performance improvements
                "speed_improvement": float(speed_improvement),
                "memory_improvement": float(memory_improvement),
                "time_improvement": float(time_improvement),
                # Baseline metrics
                "baseline_final_loss": float(baseline_avg["final_loss"]),
                "baseline_training_time": float(baseline_avg["training_time"]),
                "baseline_memory_delta": float(baseline_avg["memory_delta"]),
                "baseline_tokens_per_second": float(baseline_avg["tokens_per_second"]),
                # Evolved metrics
                "evolved_final_loss": float(evolved_avg["final_loss"]),
                "evolved_training_time": float(evolved_avg["training_time"]),
                "evolved_memory_delta": float(evolved_avg["memory_delta"]),
                "evolved_tokens_per_second": float(evolved_avg["tokens_per_second"]),
                # Trial information
                "successful_baseline_trials": comparison_results["successful_trials"]["baseline"],
                "successful_evolved_trials": comparison_results["successful_trials"]["evolved"],
                # Metadata
                "kernels_actually_used": kernels_actually_used,
            }

            # Get captured output
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()

            # Prepare simple artifacts with actual program output
            artifacts = {}
            
            if stdout_content.strip():
                artifacts["stdout"] = stdout_content.strip()
            
            if stderr_content.strip():
                artifacts["stderr"] = stderr_content.strip()

            # Add a brief execution summary
            if loss_convergence_ok and (speed_improvement > 1.1 or memory_improvement > 1.1):
                artifacts["summary"] = f"✅ Success: {speed_improvement:.2f}x speed, {memory_improvement:.2f}x memory, loss converged"
            elif loss_convergence_ok:
                artifacts["summary"] = f"✅ Loss converged but efficiency gains modest: {speed_improvement:.2f}x speed, {memory_improvement:.2f}x memory"
            else:
                artifacts["summary"] = f"❌ Loss convergence failed (diff: {loss_difference:.4f})"

            return EvaluationResult(metrics=metrics, artifacts=artifacts)

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            # Get any captured output even if there was an error
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            artifacts = {
                "stderr": error_msg + "\n" + stderr_content if stderr_content else error_msg,
                "traceback": traceback.format_exc(),
            }
            
            if stdout_content.strip():
                artifacts["stdout"] = stdout_content.strip()
            
            return EvaluationResult(
                metrics={"overall_score": 0.0, "combined_score": 0.0},
                artifacts=artifacts
            )


if __name__ == "__main__":
    print("Testing MLX LoRA Kernel Optimization Evaluator with Artifacts...")

    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")

    if os.path.exists(initial_program_path):
        result = evaluate(initial_program_path)
        print("\n=== Final Evaluation Results ===")
        print("METRICS:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        print("\nARTIFACTS:")
        for k, v in result.artifacts.items():
            print(f"  {k}: {v}")
    else:
        print(f"Initial program not found at {initial_program_path}")
