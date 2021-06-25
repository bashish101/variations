def get_cycle(choice, depth = 3):
    # choice \in [0, depth - 1]
    down = 2 ** (depth - choice)
    up = 2 ** (depth - (choice + 1))
    if choice  == 0:
        # Full-size, w0
        return [up, down, up]
    elif choice < depth:
        inner = get_cycle(choice - 1)
        
        start = [up, down]
        mid = concat(inner, inner)
        end = [down, up]
        return concat(concat(start, mid), end)
    else:
        return []

def concat(cycle1, cycle2):
    if len(cycle1) > 0 and len(cycle2) > 0:
        if cycle1[-1] == cycle2[0]:
            return cycle1 + cycle2[1:]
        # Refinement step
        return cycle1 + cycle2
    else:
        return cycle1 + cycle2

def full_multigrid(cycles = 2, depth = 3):
    start =  2 ** depth
    full_cycle = [start]
    for index in range(depth):
        w = get_cycle(index)
        sub_cycle = concat(w, w)
        full_cycle = concat(full_cycle, sub_cycle)

    grid_steps = []
    for _ in range(cycles):
        grid_steps = concat(grid_steps, full_cycle)
        
    return grid_steps
