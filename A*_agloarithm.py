import heapq
class PuzzleNode:
    def __init__(self, state, parent, move, g_cost, h_cost):
        self.state = state
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost  
    def __lt__(self, other):       
        return self.f_cost < other.f_cost
    def generate_children(self):        
        empty_pos = self.state.index('_')        
        moves = [(-3, "Up"), (3, "Down"), (-1, "Left"), (1, "Right")]
        children = []        
        for move, direction in moves:
            new_pos = empty_pos + move
            if 0 <= new_pos < 9:                
                if empty_pos % 3 == 2 and move == 1 or empty_pos % 3 == 0 and move == -1:
                    continue
                new_state = list(self.state)
                new_state[empty_pos], new_state[new_pos] = new_state[new_pos], new_state[empty_pos]
                children.append(PuzzleNode(tuple(new_state), self, direction, self.g_cost + 1, 0))
        return children
    def calculate_heuristic(self, goal_state):
        return sum(abs(b % 3 - g % 3) + abs(b // 3 - g // 3)
                   for b, g in ((self.state.index(i), goal_state.index(i)) for i in range(1, 9)))
class AStarSolver:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state
        self.open_list = []
        self.closed_list = set()
    def solve(self):
        start_node = PuzzleNode(self.start_state, None, None, 0, 0)
        start_node.h_cost = start_node.calculate_heuristic(self.goal_state)
        heapq.heappush(self.open_list, start_node)
        while self.open_list:
            current_node = heapq.heappop(self.open_list)

            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            self.closed_list.add(current_node.state)

            for child in current_node.generate_children():
                if child.state in self.closed_list:
                    continue

                child.h_cost = child.calculate_heuristic(self.goal_state)
                heapq.heappush(self.open_list, child)

        return None

    def trace_solution(self, node):
        path = []
        while node.parent is not None:
            path.append(node.move)
            node = node.parent
        return path[::-1]

    def is_solvable(self, state):
        
        flat_state = [tile for tile in state if tile != '_']
        inversions = sum(1 for i in range(len(flat_state)) for j in range(i + 1, len(flat_state)) if flat_state[i] > flat_state[j])
        return inversions % 2 == 0

start = (1, 2, '_',3, 5, 6, 4, 7, 8)  
goal = (1, 2, 3, 4, 5, 6, 7, 8, '_')  

solver = AStarSolver(start, goal)

if solver.is_solvable(start):
    solution = solver.solve()
    if solution:
        print("Solution found:", solution)
    else:
        print("No solution exists")
else:
    print("Puzzle is unsolvable")
lab 4..py
Displaying lab 4..py.
Lab-04
Qamar U Zaman
•
Oct 14
100 points
AOA,
Students here is Lab-04
Complete the tasks during Lab timings.
Thank you!

AI_Lab_04.pdf
PDF
Class comments
