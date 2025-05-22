from typing import Optional, Any
from dataclasses import dataclass
import sympy


@dataclass
class Equation:
    symbols: list
    symbol_descs: list
    symbol_properties: list
    expression: str
    desc: Optional[str] = None

    sympy_format: Optional[sympy.Expr] = None
    lambda_format: Optional[callable] = None
    program_format: Optional[str] = None

@dataclass
class SearchResult:
    equation: Equation
    aux: Any

@dataclass
class SEDTask:
    name: str
    symbols: list
    symbol_descs: list
    symbol_properties: list
    samples: Any
    desc: Optional[str] = None

@dataclass
class Problem:
    dataset_identifier: str
    equation_idx: str
    gt_equation: Equation
    samples: Any

    def create_task(self) -> SEDTask:
        return SEDTask(name=self.equation_idx,
                        symbols=self.gt_equation.symbols,
                        symbol_descs=self.gt_equation.symbol_descs,
                        symbol_properties=self.gt_equation.symbol_properties,
                        samples=self.train_samples,
                        desc=self.gt_equation.desc)
    @property
    def train_samples(self):
        return self.samples['train']
    
    @property
    def test_samples(self):
        return self.samples['test']
    
    @property
    def ood_test_samples(self):
        return self.samples.get('ood_test', None) 