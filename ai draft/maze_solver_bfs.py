"""
Name: maze_solver_bfs.py
Function description: A complete coursework solution that reads a maze from a text file, uses breadth-first
search (BFS) to compute the shortest distance from the start cell in the top-left corner to the finish
cell in the bottom-right corner, and prints the result to the screen.
Inputs: A maze text file containing the maze size and wall coordinates.
Outputs: Printed maze summary, shortest distance, and an optional visualisation of the shortest path.
Function process: The program parses the maze file, stores wall positions in a 2D grid, validates moves,
performs BFS level-by-level, reconstructs the discovered shortest path, and prints the final outcome.

Report:
This program solves a maze pathfinding problem by modelling the maze as a two-dimensional search space.
Each square in the maze is treated as a state, and movement is restricted to the four neighbouring squares:
up, down, left, and right. The starting state is fixed at the top-left corner of the maze, with coordinates
(0, 0), while the goal state is the bottom-right corner. Some squares are blocked by walls, so the algorithm
must avoid moving into them. The aim of the program is to calculate the shortest distance from the start to
the finish and display it clearly to the user.

Breadth-first search was selected because it is well suited to unweighted grid problems where each move has
the same cost. BFS explores the maze level by level, meaning it checks all positions one move away from the
start before checking positions two moves away, and so on. Because of this ordered exploration, the first
time the finish square is reached, the distance found is guaranteed to be the shortest. This makes BFS a
better choice than depth-first search for the present task.

The maze is read from a text file so that the maze size and wall positions can be changed without editing
the main program logic. The solution has been divided into separate functions to improve clarity,
maintainability, and testing. Additional validation has been included to catch invalid files, blocked start
or finish cells, and impossible mazes. The final design is efficient, readable, and easy to adapt to
alternative maze sizes for future testing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import sys


OPEN_CELL = "."
WALL_CELL = "#"
PATH_CELL = "*"
START_CELL = "S"
FINISH_CELL = "F"


@dataclass(frozen=True)
class Point:
    """A single maze coordinate represented as (row, column)."""

    row: int
    col: int


@dataclass
class Maze:
    """A container for maze data and helper properties."""

    rows: int
    cols: int
    walls: Set[Point]

    @property
    def start(self) -> Point:
        return Point(0, 0)

    @property
    def finish(self) -> Point:
        return Point(self.rows - 1, self.cols - 1)


# Name: read_maze
# Function description: Reads a maze definition from a text file and converts it into a Maze object.
# Inputs: file_path (str) - path to the maze text file.
# Outputs: Maze - an object containing the maze size and wall coordinates.
# Function process: Opens the file, ignores blank lines and comments, reads the size line, parses each wall
# coordinate, validates the data, and returns a Maze object.
def read_maze(file_path: str) -> Maze:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Maze file not found: {file_path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_lines = handle.readlines()

    cleaned_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned_lines.append(stripped)

    if not cleaned_lines:
        raise ValueError("Maze file is empty or only contains comments.")

    size_line = cleaned_lines[0].lower()
    if not size_line.startswith("size:"):
        raise ValueError("First non-comment line must be in the form: size: rows cols")

    size_part = size_line.split(":", maxsplit=1)[1].strip().replace(",", " ")
    size_tokens = size_part.split()
    if len(size_tokens) != 2:
        raise ValueError("Size line must contain exactly two integers: rows cols")

    rows, cols = map(int, size_tokens)
    if rows <= 0 or cols <= 0:
        raise ValueError("Maze dimensions must be positive integers.")

    walls: Set[Point] = set()
    for line in cleaned_lines[1:]:
        lower_line = line.lower()
        if not lower_line.startswith("wall:"):
            raise ValueError(f"Invalid line in maze file: {line}")

        coordinate_part = lower_line.split(":", maxsplit=1)[1].strip().replace(",", " ")
        tokens = coordinate_part.split()
        if len(tokens) != 2:
            raise ValueError(f"Invalid wall coordinate: {line}")

        row, col = map(int, tokens)
        wall = Point(row, col)
        if not (0 <= wall.row < rows and 0 <= wall.col < cols):
            raise ValueError(f"Wall coordinate out of bounds: ({row}, {col})")
        walls.add(wall)

    maze = Maze(rows=rows, cols=cols, walls=walls)
    if maze.start in maze.walls:
        raise ValueError("The start square (0, 0) cannot be a wall.")
    if maze.finish in maze.walls:
        raise ValueError("The finish square cannot be a wall.")

    return maze


# Name: is_safe
# Function description: Checks whether a move to a new square is valid.
# Inputs: maze (Maze) - the maze object; point (Point) - the position to test; visited (set[Point]) - already
# explored positions.
# Outputs: bool - True if the position is inside the maze, not a wall, and not visited.
# Function process: Tests the row and column boundaries, checks for a wall, then checks whether BFS has
# already explored that square.
def is_safe(maze: Maze, point: Point, visited: Set[Point]) -> bool:
    inside_maze = 0 <= point.row < maze.rows and 0 <= point.col < maze.cols
    if not inside_maze:
        return False
    if point in maze.walls:
        return False
    if point in visited:
        return False
    return True


# Name: get_neighbours
# Function description: Generates the four neighbouring squares of a given point.
# Inputs: point (Point) - the current maze position.
# Outputs: list[Point] - neighbouring coordinates in the order up, right, down, left.
# Function process: Creates new Point objects for each of the four legal movement directions.
def get_neighbours(point: Point) -> List[Point]:
    return [
        Point(point.row - 1, point.col),
        Point(point.row, point.col + 1),
        Point(point.row + 1, point.col),
        Point(point.row, point.col - 1),
    ]


# Name: bfs_shortest_distance
# Function description: Uses breadth-first search to find the shortest distance from the maze start to the
# maze finish.
# Inputs: maze (Maze) - the maze object.
# Outputs: tuple[Optional[int], list[Point]] - the shortest distance if found, otherwise None, and the path
# discovered by BFS.
# Function process: Creates a queue, explores the maze level-by-level, records parent links for path
# reconstruction, and stops as soon as the finish square is found.
def bfs_shortest_distance(maze: Maze) -> Tuple[Optional[int], List[Point]]:
    queue: Deque[Tuple[Point, int]] = deque([(maze.start, 0)])
    visited: Set[Point] = {maze.start}
    parent_map: Dict[Point, Optional[Point]] = {maze.start: None}

    while queue:
        current_point, current_distance = queue.popleft()

        if current_point == maze.finish:
            return current_distance, reconstruct_path(parent_map, maze.finish)

        for neighbour in get_neighbours(current_point):
            if is_safe(maze, neighbour, visited):
                visited.add(neighbour)
                parent_map[neighbour] = current_point
                queue.append((neighbour, current_distance + 1))

    return None, []


# Name: reconstruct_path
# Function description: Rebuilds the shortest path using parent links recorded during BFS.
# Inputs: parent_map (dict[Point, Optional[Point]]) - links from each visited node to its parent;
# finish (Point) - the final node to rebuild from.
# Outputs: list[Point] - the ordered path from the start to the finish.
# Function process: Starts at the finish node, walks backward through the parent map until the start is
# reached, then reverses the collected list.
def reconstruct_path(parent_map: Dict[Point, Optional[Point]], finish: Point) -> List[Point]:
    path: List[Point] = []
    current: Optional[Point] = finish

    while current is not None:
        path.append(current)
        current = parent_map[current]

    path.reverse()
    return path


# Name: build_display_grid
# Function description: Builds a printable version of the maze for output.
# Inputs: maze (Maze) - the maze object; path (Sequence[Point]) - the final shortest path.
# Outputs: list[list[str]] - a 2D character grid for display.
# Function process: Fills the grid with open cells, adds walls, overlays the shortest path, and finally
# marks the start and finish squares.
def build_display_grid(maze: Maze, path: Sequence[Point]) -> List[List[str]]:
    grid = [[OPEN_CELL for _ in range(maze.cols)] for _ in range(maze.rows)]

    for wall in maze.walls:
        grid[wall.row][wall.col] = WALL_CELL

    for step in path:
        if step not in {maze.start, maze.finish}:
            grid[step.row][step.col] = PATH_CELL

    grid[maze.start.row][maze.start.col] = START_CELL
    grid[maze.finish.row][maze.finish.col] = FINISH_CELL
    return grid


# Name: print_result
# Function description: Prints the maze result to the screen in a clear format.
# Inputs: maze (Maze) - the maze object; distance (Optional[int]) - shortest distance result; path
# (Sequence[Point]) - path returned from BFS.
# Outputs: None.
# Function process: Prints maze information, prints the shortest distance or a failure message, and prints
# a visual representation of the maze.
def print_result(maze: Maze, distance: Optional[int], path: Sequence[Point]) -> None:
    print(f"Maze size: {maze.rows} x {maze.cols}")
    print(f"Start: {maze.start}")
    print(f"Finish: {maze.finish}")
    print(f"Number of walls: {len(maze.walls)}")

    if distance is None:
        print("Shortest distance: No path exists from start to finish.")
        display_grid = build_display_grid(maze, [])
    else:
        print(f"Shortest distance: {distance}")
        print("Shortest path coordinates:")
        print(" -> ".join(f"({point.row},{point.col})" for point in path))
        display_grid = build_display_grid(maze, path)

    print("\nMaze view:")
    for row in display_grid:
        print(" ".join(row))


# Name: main
# Function description: Controls the overall execution of the program.
# Inputs: command-line arguments that may include a maze file path.
# Outputs: int - 0 for success or 1 for failure.
# Function process: Selects the maze file, reads the maze, runs BFS, prints the result, and handles any
# file or validation errors.
def main() -> int:
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_maze_8x8.txt"

    try:
        maze = read_maze(input_file)
        distance, path = bfs_shortest_distance(maze)
        print_result(maze, distance, path)
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
