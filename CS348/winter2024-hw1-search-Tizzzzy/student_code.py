from expand import expand
from collections import deque
import heapq

def get_path(parents, current_node):
	path = [current_node]
	while current_node in parents:
		current_node = parents[current_node]
		if current_node is not None:
			path.append(current_node)
	return path[::-1]

def a_star_search (dis_map, time_map, start, end):
	open_list = [(0, start)]
	close_list = []
	heapq.heapify(open_list)

	parents = {start: None}

	g_score = {start: 0}
	f_score = {start: dis_map[start][end]}

	while open_list:
		current = heapq.heappop(open_list)
		current_node = current[1]
		print(f"-------------------Current Node: {current_node} ------------------")
		if current_node == end:
			return get_path(parents, current_node)
		
		for neighbor in expand(current_node, time_map):
			g_cost = g_score[current_node] + time_map[current_node][neighbor]
			if g_cost < g_score.get(neighbor, float("inf")):
				parents[neighbor] = current_node
				g_score[neighbor] = g_cost
				f_score[neighbor] = g_cost + dis_map[neighbor][end]
				if neighbor not in [i[1] for i in open_list]:
					heapq.heappush(open_list, (f_score[neighbor], neighbor))
		
	return False
				


def depth_first_search(time_map, start, end):
	path = []
	visited = set()

	def dfs(node):
		if node == end:
			path.append(node)
			return True
		visited.add(node)
		path.append(node)
		for next_node in expand(node, time_map):
			if next_node not in visited and dfs(next_node):
				return True
		path.pop()
		return False
	
	if dfs(start):
		print(f"DFS path: {path}")
		return path
	else:
		return False


def breadth_first_search(time_map, start, end):
	visited = set()
	queue = deque([start])
	parents = {start: None}
	
	while queue:
		node = queue.popleft()
		if node == end:
			path = []
			while node is not None:
				path.append(node)
				node = parents[node]
			print(f"BFS path: {path}")
			# return path
			return path[::-1]
		visited.add(node)
		for next_node in expand(node, time_map):
			if next_node not in visited and next_node not in queue:
				queue.append(next_node)
				parents[next_node] = node
	return False
