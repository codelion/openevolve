"""
Test script to demonstrate the new scoring system
"""
import math

def calculate_scores(avg_time_seconds):
    """Calculate time scores for different execution times"""
    time_score = math.exp(-avg_time_seconds / 10.0)

    if avg_time_seconds < 1.0:
        time_score = min(1.0, time_score * 1.2)

    return time_score

print("=" * 70)
print("HUAWEI 2017 CDN - NEW SCORING SYSTEM")
print("=" * 70)
print("\nScoring Weights:")
print("  - Success Rate: 40%")
print("  - Cost Score:   35%")
print("  - Time Score:   25%")
print("\nFormula: combined_score = 0.40 × success_rate + 0.35 × cost_score + 0.25 × time_score")
print("\n" + "=" * 70)
print("TIME SCORE TABLE (exponential decay)")
print("=" * 70)
print(f"{'Avg Time':<12} {'Time Score':<12} {'Impact on Total':<20} {'Assessment'}")
print("-" * 70)

time_examples = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]

for t in time_examples:
    time_score = calculate_scores(t)
    impact = time_score * 0.25  # 25% weight

    if t <= 3:
        assessment = "Excellent"
    elif t <= 7:
        assessment = "Good"
    elif t <= 15:
        assessment = "Acceptable"
    elif t <= 25:
        assessment = "Poor"
    else:
        assessment = "Very Poor"

    print(f"{t:<12.1f}s {time_score:<12.4f} {impact:<20.4f} {assessment}")

print("\n" + "=" * 70)
print("COMBINED SCORE EXAMPLES")
print("=" * 70)
print("Assuming 100% success rate and perfect cost score (1.0):\n")

for t in [1, 5, 10, 20, 30]:
    time_score = calculate_scores(t)
    combined = 0.40 * 1.0 + 0.35 * 1.0 + 0.25 * time_score
    print(f"  Time {t:2d}s → combined_score = {combined:.4f}")

print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("1. Time matters significantly (25% weight)!")
print("2. Going from 5s to 10s loses 0.06 points (~6% of total score)")
print("3. Going from 10s to 20s loses 0.06 points (~6% of total score)")
print("4. Solutions slower than 20s are heavily penalized")
print("5. Target execution time: < 5 seconds per test case")
print("=" * 70)
