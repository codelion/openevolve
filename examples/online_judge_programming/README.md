# Online Judge Programming Example

This example demonstrates how OpenEvolve can solve programming problems and pass all test cases on [Kattis online judge](https://open.kattis.com/) starting from scratch.

## Problem Description

We take the [Alphabet](https://open.kattis.com/problems/alphabet) problem from [Kattis](https://open.kattis.com/) as following:
```markdown
A string of lowercase letters is called **alphabetical** if some of the letters can be deleted so that the only letters that remain are the letters from 'a' to 'z' in order. Given a string s, determine the minimum number of letters to add anywhere in the string to make it alphabetical.

Input:
Each input will consist of a single test case. Note that your program may be run multiple times on different inputs. The only line of input contains a string s (1 ≤ |s| ≤ 50) which contains only lowercase letters.

Output:
Output a single integer, which is the smallest number of letters needed to add to `s` to make it alphabetical.

Sample Input 1:
xyzabcdefghijklmnopqrstuvw

Sample Output 1:
3

Sample Input 2:
aiemckgobjfndlhp

Sample Output 2:
20
```

## Getting Started

First, fill your username and token in `example.kattisrc` according to your personal configuration file (must be logged in) from [Kattis](https://open.kattis.com/download/kattisrc) and rename the file as `.kittisrc`.

Then, to run this example:

```bash
cd examples/online_judge_programming
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

## Algorithm Evolution

### Initial Algorithm (dummy output)

The initial implementation was a simple dummy output that returned 0 directly.

```python
import sys
for line in sys.stdin:
    s = line.strip()

ans = 0
print(ans)
```

### Evolved Algorithm (Dynamic Programming)

After running OpenEvolve for just 4 iterations, it discovered a dynamic programming algorithm that passes all test cases on Kattis:

```python
import sys

for line in sys.stdin:
    s = line.strip()

n = len(s)
dp = [1] * n

for i in range(1, n):
    for j in range(i):
        if s[i] > s[j]:
            dp[i] = max(dp[i], dp[j] + 1)

longest_alphabetical_subsequence_length = max(dp)
ans = 26 - longest_alphabetical_subsequence_length
print(ans)
```

## Next Steps

Try modifying the config.yaml file to:
- Change the programming problem in system prompt
- Change the LLM model configuration
