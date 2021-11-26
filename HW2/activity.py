# 실습 2: N-Queens and Local Search
# 학번: 2016160311
# 이름: 이재윤

# todo: import statements
from nQueensTests import *

# todo: NQueens.py
qBoard = NQueens(4)
print(qBoard)
print("=================================================")

# todo: allNeighbors
neighbors = qBoard.allNeighbors()
for neigh in neighbors:
    print(neigh)

# todo: randomNeighbors
rand_neighbors = qBoard.randomNeighbors(2)
for neigh in rand_neighbors:
    print(neigh)

# todo: makeRandomMove
qBoard_rm = qBoard.makeRandomMove()
print(qBoard_rm)

# todo: crossover
qBoard2 = NQueens(4)
print(qBoard2)
qBoard_c, qBoard2_c = qBoard.crossover(qBoard2)
print(qBoard_c)
print(qBoard2_c)

# todo: other methods
print(qBoard.heuristic()) # getValue 권장
print(qBoard.getValue())
print(qBoard.getMaxValue())

# todo: hill climbing
hillClimb(qBoard)

# todo: genetic algorihm
geneticAlg(4)

# todo: test hill climbing and genetic algorithm (verbose=False)
testRandomStarts(hillClimb, reps = 5, sizeList = [4, 6])
testVaryingPops(geneticAlg, popSize=10, reps=5, sizeList=[4, 6])

# todo: 실습 과제 4번 코드 아래에 작성
# Hill-Climbing vs Stochastic Hill-Climbing
testRandomStarts(hillClimb, reps = 5, sizeList = [4, 5])
testRandomStarts(stochHillClimb, reps = 5, sizeList = [4, 5])
print("=======================================")
# Hill-Climbing vs Simulated Annealing
testRandomStarts(hillClimb, reps = 5, sizeList = [4, 5])
testRandomStarts(simAnnealing, reps = 5, sizeList = [4, 5])
print("=======================================")
# Hill-Climbing vs Beam Search
testRandomStarts(hillClimb, reps = 5, sizeList = [4, 5])
testVaryingPops(beamSearch, popSize=10, reps=5, sizeList=[4, 5])
print("=======================================")
# Genetic Algorithm vs Beam Search
testVaryingPops(geneticAlg, popSize=10, reps=5, sizeList=[4, 6])
testVaryingPops(beamSearch, popSize=10, reps=5, sizeList=[4, 6])