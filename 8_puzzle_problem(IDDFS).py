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

def dls(state, goal_state, depth, visited):
    
    if is_goal(state, goal_state):
        return [state]
    if depth == 0:
        return None
    for neighbor in get_neighbors(state):
        if tuple(neighbor) not in visited:
            visited.add(tuple(neighbor))  
            path = dls(neighbor, goal_state, depth - 1, visited)
            if path is not None:
                return [state] + path
            

    return None  

def iddfs(start_state, goal_state):
    depth = 0
    while True:
        visited = {tuple(start_state)}  
        path = dls(start_state, goal_state, depth, visited)
        if path is not None:
            return path  
        depth += 1  


start_state = [1, 2, 3, 0, 4, 6, 7, 5, 8]  
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]   

solution_path = iddfs(start_state, goal_state)

if solution_path:
    print("Solution found:")
    for state in solution_path:
        print(state)
else:
    print("No solution found.")
