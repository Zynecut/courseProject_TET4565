# TET4565: Electricity Markets and Energy System Planning - Course Project

### Overview
This project is part of the TET4565 specialization course at NTNU's Department of Electric Energy. The course focuses on electricity markets, production planning, flexibility in grid operations, and power system expansion. Students are required to work on a problem formulation of their choice, with relevance to energy markets and system planning. The project is conducted in groups of three, with each participant being responsible for understanding and explaining all parts of the analysis, results, and code.

### Project Structure
The project is divided into four main parts, each focusing on different aspects of the problem:
#### Part 1: Problem Formulation and Data Cleaning

- Objective: Define a two-stage optimization problem, perform data analysis, and clean the dataset.
- Deliverables:
  - Formulation of the optimization problem in words (max 1 page).
  - Research question(s) to be answered by the model.
  - Mathematical description of the optimization problem, including notation.
  - Code for data handling.
  - Brief description of the data, including relevant plots and summaries (max 1-2 pages).

#### Part 2: Programming a Deterministic Equivalent
- Objective: Formulate and solve the problem using a programming language, preferably Python with the Pyomo package.
- Deliverables:
  - Written document presenting the results and discussing the differences between the solutions to three problem variants (max 2 pages).
  - Code used to solve each problem variant.

#### Part 3: Benders Decomposition
- Objective: Solve the stochastic problem using Benders decomposition to handle large scenario trees.
- Deliverables:
  - Mathematical formulation of the problem using Benders decomposition.
  - Code implementing the decomposition.
  - A few paragraphs discussing and comparing the solution to the deterministic equivalent from Part 2.

#### Part 4: Stochastic Dynamic Programming
- Objective: Solve the two-stage optimization problem using stochastic dynamic programming.
- Deliverables:
- Code for solving the problem using stochastic dynamic programming.
- Summary of results and comparison to previous optimization results.
- Reflection on the solution methods and suitability for large-scale problems.

#### Example Problem: Short-Term Hydropower Optimization
- Scenario: Optimize the production profile of a hydropower station for the next 48 hours, balancing risks associated with uncertain future inflows and electricity prices.
- Input: Forecast data for the first 24 hours, with five scenarios for the remaining 24 hours.

### Important Deadlines
- Part 1: 14.09.2024 23:59
- Part 2: 26.09.2024 23:59
- Part 3: 17.10.2024 23:59
- Final Submission: 21.11.2024 23:59
