from random import shuffle

import PIL.Image
import numpy as np

algorithms = {}
import tabulate


def add_alg(func):
    """Decorator that adds the decorated function to a dictionary of algorithms.
    :param func: A function that takes a list of comparable values.
    :return: The function that was passed in.
    """
    global algorithms
    fname = func.__name__
    algorithms[fname] = func

    return func


def prepare_data(n_cells, reverse=False, shuffled=True):
    cells = list(range(n_cells))
    if reverse:
        cells.reverse()
    if shuffled:
        shuffle(cells)
    return cells


@add_alg
def heapsort(cells):
    trace = [cells[:]]
    compares = []

    def pushdown(data, start, end, trace):
        """
        Push the element in data @ start down into the maxheap
        :param data: the array being heap fixed
        :param start: the node index we start at.
        """
        root = start
        while (child := (2 * root + 1)) <= end:
            if child + 1 <= end and data[child] < data[child + 1]:
                compares.append([child, child + 1])
                child += 1
            if data[root] < data[child]:
                data[root], data[child] = data[child], data[root]
                compares.append([root, child])
                trace.append(data[:])
                root = child
            else:
                return

    # heapify
    end = len(cells) - 1
    start = (end - 1) // 2
    while start >= 0:
        pushdown(cells, start, end, trace)
        start -= 1

    while end > 0:
        cells[0], cells[end] = cells[end], cells[0]
        trace.append(cells[:])
        end -= 1
        pushdown(cells, 0, end, trace)
    return trace, compares


@add_alg
def merge(cells):
    trace = [cells[:]]
    compares = []
    n = len(cells)

    def combine(left, right, end):
        i = left
        j = right
        for k in range(left, end):
            if i < right and j < end:
                compares.append([i, j])
            if i < right and (j >= end or source[i] <= source[j]):
                dest[k] = source[i]
                i += 1
            else:
                dest[k] = source[j]
                j += 1
            trace.append(dest[:])

    source = cells[:]

    dest = cells[:]
    width = 1
    # print(f"{source=}")
    while width < n:
        i = 0
        while i < n:
            combine(i, min(i + width, n), min(i + 2 * width, n))
            i += (2 * width)
        source, dest = dest, source
        # trace.append(source[:])
        width *= 2
    return trace, compares


@add_alg
def bubble(cells):
    trace = [cells[:]]
    compares = []
    swap = True
    n = len(cells)
    while swap:
        swap = False
        for i in range(1, n):
            compares.append([i - 1, i])
            if cells[i - 1] > cells[i]:
                cells[i - 1], cells[i] = cells[i], cells[i - 1]
                swap = True
                trace.append(cells[:])
        n -= 1
        # if swap:
        #     trace.append(cells[:])
    return trace, compares


@add_alg
def insertion(cells):
    trace = [cells[:]]
    compares = []
    n = len(cells)
    i = 1
    while i < n:
        swap = False
        j = i
        compares.append([j - 1, j])
        while j > 0 and cells[j - 1] > cells[j]:
            cells[j], cells[j - 1] = cells[j - 1], cells[j]
            trace.append(cells[:])
            swap = True
            j -= 1
            if j > 0:
                compares.append([j - 1, j])
        # if swap:
        #     trace.append(cells[:])
        i += 1
    return trace, compares


@add_alg
def selection(cells):
    trace = [cells[:]]
    compares = []

    n = len(cells)
    for i in range(n):
        swap = False
        minimum_index = i
        for j in range(i + 1, n):
            compares.append([j, minimum_index])
            if cells[j] < cells[minimum_index]:
                minimum_index = j
        if i != minimum_index:
            cells[i], cells[minimum_index] = cells[minimum_index], cells[i]
            trace.append(cells[:])
            swap = True
        # if swap:
        #     trace.append(cells[:])
    return trace, compares


def render(history, destination):
    n_cells = len(history[0])
    destination.write('digraph g{ graph [rankdir="LR", ranksep=2 ];\n')
    for step, cells in enumerate(history):
        struct = "|".join(f"<f{i}> {i}" for i in cells)
        destination.write(f'"node{step}" [ label="{struct}"  shape="record"];\n')
        for i in range(1, len(history)):
            for j in range(n_cells):
                destination.write(f'"node{i - 1}":f{j} ->  "node{i}":f{j};\n')
    destination.write("}\n")


def main():
    n_cells = 256
    report = []
    start_data = prepare_data(n_cells, shuffled=True, reverse=False)
    for alg_name in ['heapsort']:
        trace, compares = algorithms[alg_name](start_data[:])
        # with open(f"{alg_name}.dot", 'w') as dest:
        #     render(trace, dest)
        memory = np.zeros((len(trace), n_cells, 3), dtype="uint8")
        for row, cols in enumerate(trace):
            cols = np.array(cols)
            memory[row, :, 0] = cols
            memory[row, :, 1] = cols
            memory[row, :, 2] = cols
        img = PIL.Image.fromarray(memory, mode="RGB")
        img = img.convert("RGB")
        img.save(f"{alg_name}_{n_cells}_memory.png")

        checks = np.zeros((len(compares), n_cells), dtype=bool)
        for row, cols in enumerate(compares):
            checks[row, cols] = 1
        img = PIL.Image.fromarray(checks)
        img.save(f"{alg_name}_{n_cells}_compares.png")
        report.append({'Algorithm': alg_name, 'Compares': len(checks), 'Assignments': len(trace)})

    print(tabulate.tabulate(report, headers='keys'))


if "__main__" == __name__:
    main()
