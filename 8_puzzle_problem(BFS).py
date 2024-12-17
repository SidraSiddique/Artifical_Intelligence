from collections import deque

def find_blank_position(state):
    return state.index(0)

def is_goal(state, goal_state):
    return state == goal_state

def get_neighbors(state):
    neighbors = []
    blank_pos = find_blank_position(state)
    moves = {
        0: [3, 1],    
        1: [0, 2, 4], 
        2: [1, 5],    
        3: [0, 4, 6], 
        4: [1, 3, 5, 7], 
        5: [2, 4],    
        6: [3, 7],    
        7: [4, 6, 8],
        8: [5, 7]     
}

    for move in moves[blank_pos]:
        new_state = state[:]
        new_state[blank_pos], new_state[move] = new_state[move], new_state[blank_pos]
        neighbors.append(new_state)

    return neighbors

def bfs(start, goal):

    queue = deque([start])

    visited = {tuple(start): None}
    
    while queue:

        current = queue.popleft()
        if is_goal(current, goal):

            path = []
            while current is not None:
                path.append(current)
                current = visited[tuple(current)]
            return path[::-1] 

        neighbors = get_neighbors(current)
        for neighbor in neighbors:
            if tuple(neighbor) not in visited:
                visited[tuple(neighbor)] = current
                queue.append(neighbor)
    
    return None 
start = [1, 2, 3, 4, 5, 6,0, 7, 8]  
goal= [1, 2, 3, 4, 5, 6, 7, 8, 0]  

solution_path = bfs(start, goal)

if solution_path:
    print("Solution found:")
    for state in solution_path:
        print(state)
else:
    print("No solution found.")
