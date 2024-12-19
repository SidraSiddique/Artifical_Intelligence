import heapq

class Node:
    def __init__(self, state, parent, move, h_cost):
        self.state = state  
        self.parent = parent  
        self.move = move  
        self.h_cost = h_cost  

    def __lt__(self, other):
        return self.h_cost < other.h_cost  

    def generate_children(self):
        children = []
        x, y = self.find_empty_tile()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:  
                new_state = [list(row) for row in self.state]  
                new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]  
                children.append(Node(new_state, self, (new_x, new_y), 0))  
        return children

    def find_empty_tile(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return (i, j)

    def calculate_heuristic(self, goal_state):
        h = 0
        for i in range(3):
            for j in range(3):
                if self.state[i][j] != 0:
                    goal_x, goal_y = divmod(self.state[i][j] - 1, 3)
                    h += abs(i - goal_x) + abs(j - goal_y)
        return h


class GreedyBestFirstSearch:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state

    def solve(self):
        start_node = Node(self.start_state, None, None, 0)
        start_node.h_cost = start_node.calculate_heuristic(self.goal_state)
        
        open_list = []
        heapq.heappush(open_list, (start_node.h_cost, start_node))
        closed_set = set()

        while open_list:
            current_node = heapq.heappop(open_list)[1]

            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            closed_set.add(tuple(map(tuple, current_node.state)))

            for child in current_node.generate_children():
                child.h_cost = child.calculate_heuristic(self.goal_state)
                if tuple(map(tuple, child.state)) not in closed_set:
                    heapq.heappush(open_list, (child.h_cost, child))

        return None  

    def trace_solution(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]  

# Example Usage
start_state = [
    [1, 2, 3],
    [4, 0, 5],
    [6, 7, 8]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

gbfs = GreedyBestFirstSearch(start_state, goal_state)
solution_path = gbfs.solve()

for step in solution_path:
    for row in step:
        print(row)
    print()
