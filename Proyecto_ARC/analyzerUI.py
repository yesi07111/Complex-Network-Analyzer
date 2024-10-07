import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
from collections import deque
from collections import Counter
import random
import time
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as patches
import re
import itertools as it
from graphviz import Graph, Digraph

class ComplexNetworkAnalyzer:
    visual_size = (11, 7)
    max_change_logs = 20

    def __init__(self):
        self.G = None
        self.pos = None

        self.change_log = deque()
    
    def save_to_change_log(self):
        tmp_G = self.G.copy()
        tmp_pos = self.pos.copy() if self.pos else None

        self.change_log.append((tmp_G, tmp_pos))
        
        if len(self.change_log) > self.max_change_logs:
            self.change_log.popleft()
    
    def revert_changes(self):
        if len(self.change_log) == 0:
            return
        self.G, self.pos = self.change_log.pop()

    def generate_graph(self, num_nodes, edge_probability, is_directed=False, is_weighted=False, 
                       is_flow_network=False, is_acyclic=False, pseudograph=False, multigraph=False, 
                       connected=False, CC_amount=1, ratio=-1, diameter=-1, weight_range=(0, 1), 
                       capacity_range=(5, 30), SCC_amount=1):

        if is_flow_network:
            
            self.G = nx.DiGraph()
            
            self.G.add_nodes_from(range(num_nodes))
    
            origin = 0
            end = num_nodes - 1
    
            # Crear un camino inicial del origen al receptor
            path = list(range(num_nodes))
            random.shuffle(path[1:-1])  # Mezclar los nodos intermedios
            for i in range(len(path) - 1):
                self.G.add_edge(path[i], path[i+1])
    
            # Añadir aristas adicionales con cierta probabilidad
            for i in range(1, num_nodes - 1):  # Excluimos el origen y el receptor
                for j in range(1, num_nodes - 1):  # Excluimos el origen y el receptor
                    if i != j and random.random() < edge_probability:
                        self.G.add_edge(i, j)
    
            # Asegurar que cada nodo tenga al menos una conexión hacia adelante y hacia atrás
            for node in range(1, num_nodes - 1):
                if self.G.out_degree(node) == 0:
                    target = random.choice(range(node + 1, num_nodes))
                    self.G.add_edge(node, target)
                if self.G.in_degree(node) == 0:
                    source = random.choice(range(0, node))
                    self.G.add_edge(source, node)
    
            for (u, v) in self.G.edges():
                self.G[u][v]['capacity'] = random.randint(capacity_range[0], capacity_range[1])
    
            self.G.nodes[origin]['label'] = 'Origin'
            self.G.nodes[end]['label'] = 'End'
    
            for pred in list(self.G.predecessors(origin)):
                self.G.remove_edge(pred, origin)
            for succ in list(self.G.successors(end)):
                self.G.remove_edge(end, succ)


        else:
            if pseudograph:
                self.G = nx.MultiGraph() if not is_directed else nx.MultiDiGraph()
            elif multigraph:
                self.G = nx.MultiGraph() if not is_directed else nx.MultiDiGraph()
            elif is_directed:
                self.G = nx.DiGraph()
            else:
                self.G = nx.Graph()

            self.G.add_nodes_from(range(num_nodes))

            # if connected:
            #     for i in range(num_nodes):
            #         for j in range(i+1, num_nodes):
            #             self.G.add_edge(i, j)
            # else:
            # nodes_per_component = num_nodes // CC_amount
            # for i in range(CC_amount):
            #     component_nodes = list(range(i*nodes_per_component, (i+1)*nodes_per_component))
            #     for u in component_nodes:
            #         for v in component_nodes:
            #             if u != v and random.random() < edge_probability:
            #                 self.G.add_edge(u, v)

            if pseudograph:
                # Añadir lazos
                for node in self.G.nodes():
                    if random.random() < edge_probability:
                        self.G.add_edge(node, node)

            if multigraph or pseudograph:
                # Añadir multiaristas
                for u, v in list(self.G.edges()):
                    if random.random() < edge_probability:
                        self.G.add_edge(u, v)

            if is_weighted:
                for u, v, key in self.G.edges(keys=True):
                    self.G[u][v][key]['weight'] = round(random.uniform(weight_range[0], weight_range[1]), 2)

            if is_acyclic:
                self.make_acyclic()

            if is_directed and not connected:
                # Asegurar SCC_amount componentes fuertemente conexas
                sccs = list(nx.strongly_connected_components(self.G))
                while len(sccs) != SCC_amount:
                    if len(sccs) < SCC_amount:
                        # Dividir una componente
                        largest_scc = max(sccs, key=len)
                        node_to_remove = random.choice(list(largest_scc))
                        self.G.remove_edges_from(list(self.G.in_edges(node_to_remove)))
                    else:
                        # Unir dos componentes
                        scc1, scc2 = random.sample(sccs, 2)
                        node1, node2 = random.choice(list(scc1)), random.choice(list(scc2))
                        self.G.add_edge(node1, node2)
                    sccs = list(nx.strongly_connected_components(self.G))

                if is_acyclic:
                    self.make_acyclic()

            if ratio != -1 or diameter != -1:
                self.adjust_radius_diameter(ratio, diameter, is_acyclic)

        return self.G

    def make_acyclic(self):
        if isinstance(self.G, (nx.MultiGraph, nx.MultiDiGraph)):
            # Para multigrafos, convertimos temporalmente a un grafo simple
            temp_G = nx.Graph() if not self.G.is_directed() else nx.DiGraph()
            temp_G.add_edges_from(self.G.edges())
            
            if temp_G.is_directed():
                # Para grafos dirigidos
                cycle_edges = list(nx.find_cycle(temp_G, orientation='original'))
                while cycle_edges:
                    edge_to_remove = random.choice(cycle_edges)
                    self.G.remove_edge(*edge_to_remove)
                    temp_G.remove_edge(*edge_to_remove)
                    try:
                        cycle_edges = list(nx.find_cycle(temp_G, orientation='original'))
                    except nx.NetworkXNoCycle:
                        break
            else:
                # Para grafos no dirigidos
                spanning_tree = nx.minimum_spanning_tree(temp_G)
                edges_to_remove = set(temp_G.edges()) - set(spanning_tree.edges())
                self.G.remove_edges_from(edges_to_remove)
        else:
            if self.G.is_directed():
                # Para grafos dirigidos
                cycle_edges = list(nx.find_cycle(self.G, orientation='original'))
                while cycle_edges:
                    edge_to_remove = random.choice(cycle_edges)
                    self.G.remove_edge(*edge_to_remove)
                    try:
                        cycle_edges = list(nx.find_cycle(self.G, orientation='original'))
                    except nx.NetworkXNoCycle:
                        break
            else:
                # Para grafos no dirigidos
                spanning_tree = nx.minimum_spanning_tree(self.G)
                self.G = spanning_tree

    def adjust_radius_diameter(self, target_radius, target_diameter, is_acyclic):
        while True:
            current_radius = nx.radius(self.G)
            current_diameter = nx.diameter(self.G)
            
            if (target_radius == -1 or current_radius == target_radius) and \
               (target_diameter == -1 or current_diameter == target_diameter):
                break
            
            if current_radius < target_radius or current_diameter < target_diameter:
                # Necesitamos aumentar el radio o el diámetro
                u, v = self.find_distant_nodes()
                if not self.G.has_edge(u, v):
                    if not is_acyclic or not self.will_create_cycle(u, v):
                        self.G.add_edge(u, v)
            else:
                # Necesitamos disminuir el radio o el diámetro
                edges = list(self.G.edges())
                if edges:
                    u, v = random.choice(edges)
                    if nx.number_of_edges(self.G) > nx.number_of_nodes(self.G) - 1:
                        self.G.remove_edge(u, v)

    def find_distant_nodes(self):
        # Encuentra dos nodos que estén a la mayor distancia posible
        nodes = list(self.G.nodes())
        u = random.choice(nodes)
        distances = nx.single_source_shortest_path_length(self.G, u)
        v = max(distances, key=distances.get)
        return u, v

    def will_create_cycle(self, u, v):
        # Verifica si añadir una arista entre u y v creará un ciclo
        return nx.has_path(self.G, u, v)


    def analyze_information_distribution(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        degree_sequence = [d for n, d in self.G.degree()]
        fig, ax = plt.subplots(figsize=self.visual_size)
        ax.hist(degree_sequence, bins=50, edgecolor='black')
        ax.set_title("Distribución de grado en grafo de 5000 nodos")
        ax.set_xlabel("Grado")
        ax.set_ylabel("Frecuencia")
        
        avg_degree = np.mean(degree_sequence)
        
        return fig, avg_degree

    def calculate_centrality(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        centrality = nx.degree_centrality(self.G)
        return centrality
    # 

    def find_shortest_path(self, start, end):
        if self.G is None:
            return "Grafo no generado aún"
        
        try:
            return nx.shortest_path(self.G, start, end)
        except nx.NetworkXNoPath:
            return f"No hay camino de {start} a {end}"

    def shortest_weighted_path(self, start, end):
        if self.G is None:
            return "Grafo no generado aún"
        
        try:
            path = nx.shortest_path(self.G, start, end, weight='weight')
            cost = sum(self.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            return path, cost
        except nx.NetworkXNoPath:
            return "No existe un camino entre los nodos especificados", None
    
    def calculate_diameter(self):
        if self.G is None:
            return "Grafo no generado aún"

        if nx.is_connected(self.G):
            return nx.diameter(self.G)
        else:
            return "El grafo no es conexo, el diámetro no está definido"

    def calculate_radius(self):
        if self.G is None:
            return "Grafo no generado aún"

        if nx.is_connected(self.G):
            return nx.radius(self.G)
        else:
            return "El grafo no es conexo, el radio no está definido"

    def find_longest_path(self, start, end, max_iterations=1000, time_limit=10):
        if self.G is None:
            return "Grafo no generado aún"

        def calculate_path_length(path):
            return len(path) - 1

        longest_path = None
        path_length = 0
        iterations = 0
        start_time = time.time()

        try:
            for path in nx.all_simple_paths(self.G, start, end):
                iterations += 1
                current_length = calculate_path_length(path)
                if current_length > path_length:
                    longest_path = path
                    path_length = current_length

                if iterations >= max_iterations or time.time() - start_time > time_limit:
                    break

        except nx.NetworkXNoPath:
            return f"No hay camino entre {start} y {end}"

        if longest_path is None:
            return f"No se encontró un camino entre {start} y {end}"

        return (f"El camino más largo encontrado después de {iterations} iteraciones es {longest_path} con longitud {path_length}", longest_path, path_length)

    def find_max_cost_path(self, start, end, max_iterations=1000, time_limit=10):
        if self.G is None:
            return "Grafo no generado aún"

        def calculate_path_cost(path):
            return sum(self.G[path[i]][path[i+1]].get('weight', 1) for i in range(len(path)-1))

        max_cost_path = None
        max_cost = float('-inf')
        iterations = 0
        start_time = time.time()

        try:
            for path in nx.all_simple_paths(self.G, start, end):
                iterations += 1
                current_cost = calculate_path_cost(path)
                if current_cost > max_cost:
                    max_cost_path = path
                    max_cost = current_cost

                if iterations >= max_iterations or time.time() - start_time > time_limit:
                    break

        except nx.NetworkXNoPath:
            return f"No hay camino entre {start} y {end}"

        if max_cost_path is None:
            return f"No se encontró un camino entre {start} y {end}"

        return f"El camino de costo máximo encontrado después de {iterations} iteraciones es {max_cost_path} con costo {max_cost}"

    def find_global_min_cost_path(self, max_iterations=1000, time_limit=10):
        if self.G is None:
            return "Grafo no generado aún"

        if not nx.is_connected(self.G):
            return "El grafo no es conexo, no se puede calcular el camino de costo mínimo global"

        min_cost = float('inf')
        min_path = None
        iterations = 0
        start_time = time.time()

        nodes = list(self.G.nodes())
        while iterations < max_iterations and time.time() - start_time <= time_limit:
            start = random.choice(nodes)
            end = random.choice(nodes)
            if start != end:
                try:
                    path = nx.shortest_path(self.G, start, end, weight='weight')
                    cost = nx.path_weight(self.G, path, weight='weight')
                    if cost < min_cost:
                        min_cost = cost
                        min_path = path
                except nx.NetworkXNoPath:
                    pass
            iterations += 1

        if min_path is None:
            return "No se encontró ningún camino en el grafo"

        return f"El camino de costo mínimo global encontrado después de {iterations} iteraciones es {min_path} con costo {min_cost}"

    def find_global_max_cost_path(self, max_iterations=1000, time_limit=10):
        if self.G is None:
            return "Grafo no generado aún"

        max_cost = float('-inf')
        max_path = None
        iterations = 0
        start_time = time.time()

        nodes = list(self.G.nodes())
        while iterations < max_iterations and time.time() - start_time <= time_limit:
            start = random.choice(nodes)
            end = random.choice(nodes)
            if start != end:
                try:
                    for path in nx.all_simple_paths(self.G, start, end):
                        cost = sum(self.G[path[i]][path[i+1]].get('weight', 1) for i in range(len(path)-1))
                        if cost > max_cost:
                            max_cost = cost
                            max_path = path
                except nx.NetworkXNoPath:
                    pass
            iterations += 1

        if max_path is None:
            return "No se encontró ningún camino en el grafo"

        return f"El camino de costo máximo global encontrado después de {iterations} iteraciones es {max_path} con costo {max_cost}"

    def clustering_coefficient(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        local_clustering = nx.clustering(self.G)
        average_clustering = nx.average_clustering(self.G)
        
        return local_clustering, average_clustering

    def average_shortest_path(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        try:
            if nx.is_directed(self.G):
                # Para grafos dirigidos
                if nx.is_strongly_connected(self.G):
                    return nx.average_shortest_path_length(self.G)
                else:
                    # Calcular para cada componente fuertemente conexa
                    components = list(nx.strongly_connected_components(self.G))
                    if len(components) == 1:
                        return None  # No se puede calcular si solo hay una componente
                    avg_paths = []
                    for component in components:
                        subgraph = self.G.subgraph(component)
                        if len(subgraph) > 1:
                            avg_paths.append(nx.average_shortest_path_length(subgraph))
                    return sum(avg_paths) / len(avg_paths) if avg_paths else None
            else:
                # Para grafos no dirigidos
                if nx.is_connected(self.G):
                    return nx.average_shortest_path_length(self.G)
                else:
                    # Calcular para cada componente conexa
                    components = list(nx.connected_components(self.G))
                    if len(components) == 1:
                        return None  # No se puede calcular si solo hay una componente
                    avg_paths = []
                    for component in components:
                        subgraph = self.G.subgraph(component)
                        if len(subgraph) > 1:
                            avg_paths.append(nx.average_shortest_path_length(subgraph))
                    return sum(avg_paths) / len(avg_paths) if avg_paths else None
        except Exception as e:
            return f"Error al calcular el camino más corto promedio: {str(e)}"

    def betweenness_centrality(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        betweenness = nx.betweenness_centrality(self.G)
        return betweenness
# 
    def minimum_cost_path(self, start, end):
        if self.G is None:
            return "Grafo no generado aún"
        
        if not nx.is_directed_acyclic_graph(self.G):
            return "El grafo debe ser dirigido y acíclico para este método"
        
        if 'weight' not in self.G.edges[next(iter(self.G.edges()))]:
            return "El grafo debe tener pesos en las aristas"
        
        # Ordenamiento topológico
        topological_order = list(nx.topological_sort(self.G))
        
        # Inicialización
        dist = {node: float('inf') for node in self.G.nodes()}
        dist[start] = 0
        prev = {node: None for node in self.G.nodes()}
        
        # Relajación de aristas
        for u in topological_order:
            for v in self.G.successors(u):
                if dist[v] > dist[u] + self.G[u][v]['weight']:
                    dist[v] = dist[u] + self.G[u][v]['weight']
                    prev[v] = u
        
        # Reconstrucción del camino
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        if path[0] != start:
            return f"No hay camino de {start} a {end}"
        
        return f"Camino de costo mínimo de {start} a {end}: {path}, Costo: {dist[end]}"

    def _bfs_augmenting_path(self, graph, source, sink):
        queue = deque([(source, [source])])
        visited = set([source])
        
        while queue:
            (node, path) = queue.popleft()
            for next_node in graph[node]:
                if graph[node][next_node]['capacity'] > 0 and next_node not in visited:
                    if next_node == sink:
                        return path + [next_node]
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return None
    
    def assortativity(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.degree_assortativity_coefficient(self.G)

    def eigenvector_centrality(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.eigenvector_centrality(self.G)

    def community_detection(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return list(nx.community.greedy_modularity_communities(self.G))

    def small_world_coefficient(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        random_graph = nx.erdos_renyi_graph(self.G.number_of_nodes(), nx.density(self.G))
        
        C = nx.average_clustering(self.G)
        C_rand = nx.average_clustering(random_graph)
        
        L = nx.average_shortest_path_length(self.G)
        L_rand = nx.average_shortest_path_length(random_graph)
        
        sigma = (C / C_rand) / (L / L_rand)
        
        return sigma

    def scale_free_test(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        degrees = [d for n, d in self.G.degree()]
        degree_counts = Counter(degrees)
        x = list(degree_counts.keys())
        y = list(degree_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.loglog(x, y, 'bo')
        plt.xlabel('Grado (log)')
        plt.ylabel('Frecuencia (log)')
        plt.title('Distribución de grado (escala log-log)')
        plt.show()
        
        # Ajuste de ley de potencia
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(x), np.log(y))
        
        return f"Coeficiente de la ley de potencia: {-slope:.2f}, R-squared: {r_value**2:.2f}"

    def bridge_detection(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return list(nx.bridges(self.G))

    def articulation_points(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return list(nx.articulation_points(self.G))

    def rich_club_coefficient(self, normalized=True):
        if self.G is None:
            return "Grafo no generado aún"
        
        try:
            rc = nx.rich_club_coefficient(self.G, normalized=normalized)
            # Filtrar los valores que son None o infinito
            rc = {k: v for k, v in rc.items() if v is not None and not np.isinf(v)}
            return rc
        except ZeroDivisionError:
            print("No se pudo calcular el coeficiente de Rich Club normalizado para algunos grados.")
            # Intentar calcular el coeficiente no normalizado
            return nx.rich_club_coefficient(self.G, normalized=False)

    def degree_distribution(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        degrees = [d for n, d in self.G.degree()]
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1
        
        total_nodes = self.G.number_of_nodes()
        degree_prob = {k: v/total_nodes for k, v in degree_counts.items()}
        
        plt.figure(figsize=(10, 6))
        plt.bar(degree_prob.keys(), degree_prob.values())
        plt.xlabel('Grado')
        plt.ylabel('Probabilidad')
        plt.title('Distribución de grado')
        plt.show()
        
        return degree_prob

    def average_neighbor_degree(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.average_neighbor_degree(self.G)

    def core_number(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.core_number(self.G)
    
    def approximate_clique_number(self):
        if self.G is None:
            return "Grafo no generado aún"

        import networkx as nx

        # Usaremos el número de core máximo como una aproximación del número de clique
        core_numbers = nx.core_number(self.G)
        approx_clique_number = max(core_numbers.values())

        return approx_clique_number
    
    def global_efficiency(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.global_efficiency(self.G)

    def local_efficiency(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.local_efficiency(self.G)

    def spectral_gap(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        laplacian_spectrum = nx.laplacian_spectrum(self.G)
        sorted_spectrum = sorted(laplacian_spectrum, reverse=True)
        return sorted_spectrum[0] - sorted_spectrum[1]

    def degree_pearson_correlation_coefficient(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        return nx.degree_pearson_correlation_coefficient(self.G)

    def average_shortest_path_length(self):
        if self.G is None:
            return "Grafo no generado aún"

        try:
            return nx.average_shortest_path_length(self.G)
        except nx.NetworkXError:
            components = list(nx.connected_components(self.G))
            avg_paths = []
            for component in components:
                subgraph = self.G.subgraph(component)
                if len(subgraph) > 1:
                    avg_paths.append(nx.average_shortest_path_length(subgraph))

            if avg_paths:
                return np.mean(avg_paths)
            else:
                return None

    def diameter_of_forest(self):
        if self.G is None:
            return "Grafo no generado aún"

        import networkx as nx

        components = list(nx.connected_components(self.G))
        max_diameter = 0
        for component in components:
            subgraph = self.G.subgraph(component)
            diameter = nx.diameter(subgraph)
            if diameter > max_diameter:
                max_diameter = diameter

        return max_diameter

    def centers_of_forest(self):
        if self.G is None:
            return "Grafo no generado aún"

        components = list(nx.connected_components(self.G))
        centers = []
        for component in components:
            subgraph = self.G.subgraph(component)
            centers.append(nx.center(subgraph))

        return centers
    
    def centroids_of_forest(self):
        if self.G is None:
            return "Grafo no generado aún"

        import networkx as nx

        def find_centroid(tree):
            tree = tree.copy()  # Trabajamos con una copia del árbol
            n = len(tree)
            if n == 1:
                return list(tree.nodes)[0]

            while len(tree) > 2:
                leaves = [node for node in tree.nodes if tree.degree(node) == 1]
                tree.remove_nodes_from(leaves)

            return list(tree.nodes)

        components = list(nx.connected_components(self.G))
        centroids = []
        for component in components:
            subgraph = self.G.subgraph(component).copy()  # Creamos una copia del subgrafo
            centroids.append(find_centroid(subgraph))

        return centroids
    
    def leaves_count_of_forest(self):
        if self.G is None:
            return "Grafo no generado aún"

        components = list(nx.connected_components(self.G))
        leaf_counts = []
        for component in components:
            subgraph = self.G.subgraph(component)
            leaves = sum(1 for node in subgraph.nodes if subgraph.degree(node) == 1)
            leaf_counts.append(leaves)

        return leaf_counts

    def heights_of_forest(self):
        if self.G is None:
            return "Grafo no generado aún"
        
        def tree_height(tree):
            if len(tree) == 1:
                return 0
            root = list(tree.nodes)[0]  # Elegimos un nodo arbitrario como raíz
            return max(nx.shortest_path_length(tree, root).values())

        components = list(nx.connected_components(self.G))
        heights = []
        for component in components:
            subgraph = self.G.subgraph(component)
            heights.append(tree_height(subgraph))

        return heights

    def pagerank_centrality(self):
        if self.G is None:
            return "Grafo no generado aún"

        pagerank = nx.pagerank(self.G)
        return pagerank
    
    def katz_centrality(self):
        if self.G is None:
            return "Grafo no generado aún"

        katz = nx.katz_centrality(self.G)
        return katz
    
    def shortest_path_dag(self, start, end):
        if self.G is None:
            return "Grafo no generado aún"
        
        if not nx.is_directed_acyclic_graph(self.G):
            return "El grafo no es acíclico dirigido"
        
        try:
            path = nx.shortest_path(self.G, start, end, weight='weight')
            cost = sum(self.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            return path, cost
        except nx.NetworkXNoPath:
            return f"No existe un camino de {start} a {end}", None

    def topological_sort(self):
        if self.G is None:
            return "Grafo no generado aún"

        if nx.is_directed_acyclic_graph(self.G):
            return list(nx.topological_sort(self.G))
        else:
            return "El grafo no es acíclico dirigido"
    
    def count_paths(self, start, end):
        if self.G is None:
            return "Grafo no generado aún"

        if not nx.is_directed_acyclic_graph(self.G):
            return "El grafo no es acíclico dirigido"

        def dfs_count(node):
            if node == end:
                return 1
            if node not in memo:
                memo[node] = sum(dfs_count(succ) for succ in self.G.successors(node))
            return memo[node]

        memo = {}
        return dfs_count(start)
    
    def transitive_reduction(self):
        if self.G is None:
            return "Grafo no generado aún"

        if nx.is_directed_acyclic_graph(self.G):
            reduced = nx.transitive_reduction(self.G)
            original_edges = self.G.number_of_edges()
            reduced_edges = reduced.number_of_edges()
            return f"Aristas originales: {original_edges}\nAristas después de la reducción: {reduced_edges}"
        else:
            return "El grafo no es acíclico dirigido"

    def max_flow(self, source, sink):
        if self.G is None:
            return "Grafo no generado aún"

        # Calcula el flujo máximo
        flow_value, flow_dict = nx.maximum_flow(self.G, source, sink)

        # Crea un nuevo grafo con la información del flujo
        flow_graph = self.G.copy()
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                flow_graph[u][v]['flow'] = flow

        return flow_value, flow_graph

    def visualize_flow(self, flow_graph, source, sink):
        plt.figure(figsize=(40, 40))
        pos = nx.spring_layout(flow_graph, k=3, iterations=50)
        
        nx.draw(flow_graph, pos, node_size=3000, node_color='lightpink', with_labels=False, arrows=True, arrowsize=40)

        labels = {node: flow_graph.nodes[node].get('label', str(node)) for node in flow_graph.nodes()}
        nx.draw_networkx_labels(flow_graph, pos, labels, font_size=24, font_weight='bold')

        edge_labels = {}
        for u, v, data in flow_graph.edges(data=True):
            flow = data.get('flow', 0)
            capacity = data['capacity']
            edge_labels[(u, v)] = f"{flow}/{capacity}"

        nx.draw_networkx_edge_labels(flow_graph, pos, edge_labels=edge_labels, font_size=20)

        # Resalta el origen y el sumidero
        nx.draw_networkx_nodes(flow_graph, pos, nodelist=[source], node_color='lightgreen', node_size=3500)
        nx.draw_networkx_nodes(flow_graph, pos, nodelist=[sink], node_color='lightblue', node_size=3500)

        plt.title("Red de Flujo Máximo", fontsize=80)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class NetworkAnalyzerGUI:
    
    visual_size = (11, 7)

    def __init__(self, master):
        self.master = master
        self.master.title("Complex Network Analyzer")
        self.analyzer = ComplexNetworkAnalyzer()

        # Configurar el grid del master
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=10)  # Dar más espacio a la columna de la imagen
        for i in range(3):
            master.rowconfigure(i, weight=1)

        # Frame para la generación del grafo
        self.gen_frame = ttk.LabelFrame(master, text="Generación del Grafo")
        self.gen_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Frame para la edición del grafo
        self.edit_frame = ttk.LabelFrame(master, text="Edición del Grafo")
        self.edit_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Frame para el análisis del grafo
        self.analysis_frame = ttk.LabelFrame(master, text="Análisis del Grafo")
        self.analysis_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Frame para la visualización
        self.plot_frame = ttk.LabelFrame(master, text="Visualización")
        self.plot_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        self.canvas = None

        #region generacion
        # Frame para la generación del grafo
        self.gen_frame = ttk.LabelFrame(master, text="Generación del Grafo")
        self.gen_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Opciones básicas
        ttk.Label(self.gen_frame, text="Número de nodos:").grid(row=0, column=0, sticky="w")
        self.num_nodes = tk.IntVar(value=100)
        ttk.Entry(self.gen_frame, textvariable=self.num_nodes).grid(row=0, column=1)

        ttk.Label(self.gen_frame, text="Probabilidad de arista:").grid(row=1, column=0, sticky="w")
        self.edge_prob = tk.DoubleVar(value=0.1)
        ttk.Entry(self.gen_frame, textvariable=self.edge_prob).grid(row=1, column=1)

        # Tipo de grafo
        ttk.Label(self.gen_frame, text="Tipo de grafo:").grid(row=2, column=0, sticky="w")
        self.graph_type = tk.StringVar()
        graph_types = ["Red de flujo", "Pseudografo", "Multigrafo", "Grafo"]
        self.graph_type_menu = ttk.Combobox(self.gen_frame, textvariable=self.graph_type, values=graph_types)
        self.graph_type_menu.grid(row=2, column=1)
        self.graph_type_menu.bind("<<ComboboxSelected>>", self.update_options)

        # Opciones adicionales (inicialmente ocultas)
        self.additional_options = {}

        # Opciones 
        # Opciones para ponderación
        self.additional_options["weight_type"] = tk.StringVar(value="Racional")
        self.weight_type_label = ttk.Label(self.gen_frame, text="Tipo de peso:")
        self.weight_type_menu = ttk.Combobox(self.gen_frame, textvariable=self.additional_options["weight_type"], values=["Entero", "Racional"])
        self.weight_type_menu.bind("<<ComboboxSelected>>", self.update_decimal_places)

        self.additional_options["decimal_places"] = tk.IntVar(value=2)
        self.decimal_places_label = ttk.Label(self.gen_frame, text="Lugares después de la coma:")
        self.decimal_places_entry = ttk.Entry(self.gen_frame, textvariable=self.additional_options["decimal_places"])

        # Opciones para red de flujo
        self.additional_options["capacity_type"] = tk.StringVar(value="Entero")
        self.capacity_type_label = ttk.Label(self.gen_frame, text="Tipo de capacidad:")
        self.capacity_type_menu = ttk.Combobox(self.gen_frame, textvariable=self.additional_options["capacity_type"], values=["Entero", "Racional"])
        self.capacity_type_menu.bind("<<ComboboxSelected>>", self.update_capacity_decimal_places)

        self.additional_options["capacity_decimal_places"] = tk.IntVar(value=2)
        self.capacity_decimal_places_label = ttk.Label(self.gen_frame, text="Lugares después de la coma (capacidad):")
        self.capacity_decimal_places_entry = ttk.Entry(self.gen_frame, textvariable=self.additional_options["capacity_decimal_places"])
        
        self.additional_options["capacity_range"] = tk.BooleanVar()
        self.capacity_check = ttk.Checkbutton(self.gen_frame, text="Especificar rango de capacidades", variable=self.additional_options["capacity_range"], command=self.update_capacity_range)
        self.min_capacity = tk.IntVar(value=5)
        self.max_capacity = tk.IntVar(value=30)
        self.min_capacity_label = ttk.Label(self.gen_frame, text="Min capacidad:")
        self.min_capacity_entry = ttk.Entry(self.gen_frame, textvariable=self.min_capacity)
        self.max_capacity_label = ttk.Label(self.gen_frame, text="Max capacidad:")
        self.max_capacity_entry = ttk.Entry(self.gen_frame, textvariable=self.max_capacity)

        self.additional_options["is_weighted"] = tk.BooleanVar()
        self.weighted_check = ttk.Checkbutton(self.gen_frame, text="Ponderado", variable=self.additional_options["is_weighted"], command=self.update_weight_options)

        self.additional_options["weight_type"] = tk.StringVar(value="Racional")
        self.weight_type_label = ttk.Label(self.gen_frame, text="Tipo de peso:")
        self.weight_type_menu = ttk.Combobox(self.gen_frame, textvariable=self.additional_options["weight_type"], values=["Entero", "Racional"], state="readonly")
        self.weight_type_menu.bind("<<ComboboxSelected>>", self.update_weight_options)

        self.additional_options["decimal_places"] = tk.IntVar(value=2)
        self.decimal_places_label = ttk.Label(self.gen_frame, text="Lugares después de la coma:")
        self.decimal_places_entry = ttk.Entry(self.gen_frame, textvariable=self.additional_options["decimal_places"], width=5)

        self.additional_options["weight_range"] = tk.BooleanVar()
        self.weight_check = ttk.Checkbutton(self.gen_frame, text="Especificar rango de pesos", variable=self.additional_options["weight_range"], command=self.update_weight_options)
        self.min_weight = tk.DoubleVar(value=0.0)
        self.max_weight = tk.DoubleVar(value=1.0)
        self.min_weight_label = ttk.Label(self.gen_frame, text="Min peso:")
        self.min_weight_entry = ttk.Entry(self.gen_frame, textvariable=self.min_weight)
        self.max_weight_label = ttk.Label(self.gen_frame, text="Max peso:")
        self.max_weight_entry = ttk.Entry(self.gen_frame, textvariable=self.max_weight)

        
        self.additional_options["is_acyclic"] = tk.BooleanVar()
        self.acyclic_check = ttk.Checkbutton(self.gen_frame, text="Acíclico", variable=self.additional_options["is_acyclic"])
        self.pseudo_acyclic_check = ttk.Checkbutton(self.gen_frame, text="Pseudo Acíclico", variable=self.additional_options["is_acyclic"])
        
        self.additional_options["is_directed"] = tk.BooleanVar()
        self.directed_check = ttk.Checkbutton(self.gen_frame, text="Dirigido", variable=self.additional_options["is_directed"])

        self.layout_option = tk.StringVar(value="Resorte")
        self.node_distance = tk.IntVar(value=1)
        
        ttk.Button(self.gen_frame, text="Generar Grafo", command=self.generate_graph).grid(row=15, column=0)
        ttk.Button(self.gen_frame, text="Opciones de Visualización", command=self.show_visualization_options).grid(row=15, column=1)

        self.show_created_graph = ttk.Button(self.gen_frame, text="Mostrar Grafo", command=self.show_graph)
        self.show_induced_graph = ttk.Button(self.gen_frame, text="Subgrafo Inducido", command=self.induced_subgraph)
        self.show_created_graph.grid(row=16, column=0)
        self.show_induced_graph.grid(row=16, column=1) 
        self.show_created_graph.grid_remove()
        self.show_induced_graph.grid_remove()

        self.node_weight_see = ttk.Button(self.gen_frame, text="Ver peso/capacidad de aristas de 1 nodo", command=self.update_node_weight_see)
        self.node_weight_see.grid(row=17, column=0)
        self.node_weight_see.grid_remove()

        self.two_nodes_weight_see = ttk.Button(self.gen_frame, text="Ver peso/capacidad aristas entre dos nodos", command=self.update_two_nodes_weight_see)
        self.two_nodes_weight_see.grid(row=17, column=1)
        self.two_nodes_weight_see.grid_remove()

        #endregion

        #region edicion

        ttk.Button(self.edit_frame, text="Añadir nodo", command=self.add_node).grid(row=3, column=0, pady=5)
        ttk.Button(self.edit_frame, text="Añadir arista", command=self.add_edge).grid(row=3, column=1, pady=5)
        ttk.Button(self.edit_frame, text="Eliminar nodo", command=self.remove_node).grid(row=4, column=0, pady=5)
        ttk.Button(self.edit_frame, text="Eliminar arista", command=self.remove_edge).grid(row=4, column=1, pady=5)
        ttk.Button(self.edit_frame, text="Deshacer ultimo cambio", command=self.undo).grid(row=5, column=0, pady=5)

        self.modify_edge_button = ttk.Button(self.edit_frame, text="Modificar arista", command=self.modify_edge)
        self.modify_edge_button.grid(row=5, column=1, pady=5)
        self.modify_edge_button.grid_remove()  # Inicialmente oculto
        #endregion

        #region analisis

        ttk.Button(self.analysis_frame, text="Distribución de Información", command=self.analyze_information_distribution).grid(row=0, column=0)
        ttk.Button(self.analysis_frame, text="Calcular Centralidad", command=self.calculate_centrality).grid(row=0, column=1)

        ttk.Button(self.analysis_frame, text="Camino más largo", command=self.show_longest_path).grid(row=1, column=0)
        ttk.Button(self.analysis_frame, text="Camino más corto", command=self.show_shortest_path).grid(row=1, column=1)
        ttk.Button(self.analysis_frame, text="Camino de costo mínimo", command=self.show_shortest_weighted_path).grid(row=2, column=0)
        ttk.Button(self.analysis_frame, text="Coeficiente de agrupamiento", command=self.show_clustering_coefficient).grid(row=2, column=1)
        ttk.Button(self.analysis_frame, text="Camino más corto promedio", command=self.show_average_shortest_path).grid(row=3, column=0)
        ttk.Button(self.analysis_frame, text="Centralidad de intermediación", command=self.show_betweenness_centrality).grid(row=3, column=1)
        #endregion

        self.canvas = None
    
    def show_visualization_options(self):
        options_window = tk.Toplevel(self.master)
        options_window.title("Opciones de Visualización")
        ttk.Label(options_window, text="Visualización:").grid(row=0, column=0, padx=5, pady=5)
        layout_dropdown = ttk.Combobox(options_window, textvariable=self.layout_option, 
                                    values=["Resorte", "Circular", "Aleatorio", "Bonito"])
        layout_dropdown.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(options_window, text="Distancia entre nodos:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(options_window, textvariable=self.node_distance).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(options_window, text="Aplicar", command=options_window.destroy).grid(row=2, column=0, columnspan=2, pady=10)

    def undo(self):
        self.analyzer.revert_changes()
        self.show_graph()
    
    def update_options(self, event):
        # Ocultar todas las opciones adicionales
        for widget in [self.capacity_check, self.min_capacity_label, self.min_capacity_entry,
                    self.max_capacity_label, self.max_capacity_entry, self.weighted_check,
                    self.acyclic_check, self.pseudo_acyclic_check, self.directed_check,
                    self.weight_type_label, self.weight_type_menu, self.decimal_places_label,
                    self.decimal_places_entry, self.capacity_type_label, self.capacity_type_menu,
                    self.capacity_decimal_places_label, self.capacity_decimal_places_entry,
                    self.weight_check, self.min_weight_label, self.min_weight_entry,
                    self.max_weight_label, self.max_weight_entry]:
            widget.grid_remove()

        # Mostrar opciones según el tipo de grafo seleccionado
        selected_type = self.graph_type.get()
        row = 3
        if selected_type == "Red de flujo":
            self.capacity_check.grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1
            self.capacity_type_label.grid(row=row, column=0, sticky="w")
            self.capacity_type_menu.grid(row=row, column=1, sticky="w")
            row += 1
            self.update_capacity_options()
        else:  # Para Grafo, Pseudografo y Multigrafo
            self.directed_check.grid(row=row, column=0, sticky="w")
            row += 1

            if selected_type == "Pseudografo":
                self.pseudo_acyclic_check.grid(row=row, column=0, sticky="w")
            else:
                self.acyclic_check.grid(row=row, column=0, sticky="w")
            row += 1

            self.weighted_check.grid(row=row, column=0, sticky="w")
            row += 1
            self.update_weight_options()

        self.update_modify_edge_btn()

    def update_weight_options(self, event=None):
        if self.additional_options["is_weighted"].get():
            row = self.weighted_check.grid_info()['row'] + 1
            self.weight_type_label.grid(row=row, column=0, sticky="w")
            self.weight_type_menu.grid(row=row, column=1, sticky="w")
            if self.analyzer.G:
                self.node_weight_see.grid(row=17, column=0)
                self.two_nodes_weight_see.grid(row=17, column=1)
            row += 1

            if self.additional_options["weight_type"].get() == "Racional":
                self.decimal_places_label.grid(row=row, column=0, sticky="w")
                self.decimal_places_entry.grid(row=row, column=1, sticky="w")
                row += 1
            else:
                self.decimal_places_label.grid_remove()
                self.decimal_places_entry.grid_remove()

            self.weight_check.grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1

            if self.additional_options["weight_range"].get():
                self.min_weight_label.grid(row=row, column=0, sticky="w")
                self.min_weight_entry.grid(row=row, column=1, sticky="w")
                row += 1
                self.max_weight_label.grid(row=row, column=0, sticky="w")
                self.max_weight_entry.grid(row=row, column=1, sticky="w")
            else:
                self.min_weight_label.grid_remove()
                self.min_weight_entry.grid_remove()
                self.max_weight_label.grid_remove()
                self.max_weight_entry.grid_remove()
        else:
            self.weight_type_label.grid_remove()
            self.weight_type_menu.grid_remove()
            self.decimal_places_label.grid_remove()
            self.decimal_places_entry.grid_remove()
            self.weight_check.grid_remove()
            self.min_weight_label.grid_remove()
            self.min_weight_entry.grid_remove()
            self.max_weight_label.grid_remove()
            self.max_weight_entry.grid_remove()
            self.node_weight_see.grid_remove()
            self.two_nodes_weight_see.grid_remove()

    def update_capacity_options(self, event=None):
        row = self.capacity_type_menu.grid_info()['row'] + 1
        capacity_type = self.additional_options["capacity_type"].get()
        
        if capacity_type == "Racional":
            self.capacity_decimal_places_label.grid(row=row, column=0, sticky="w")
            self.capacity_decimal_places_entry.grid(row=row, column=1, sticky="w")
            row += 1
        else:
            self.capacity_decimal_places_label.grid_remove()
            self.capacity_decimal_places_entry.grid_remove()
        
        if self.additional_options["capacity_range"].get():
            self.min_capacity_label.grid(row=row, column=0, sticky="w")
            self.min_capacity_entry.grid(row=row, column=1, sticky="w")
            row += 1
            self.max_capacity_label.grid(row=row, column=0, sticky="w")
            self.max_capacity_entry.grid(row=row, column=1, sticky="w")
        else:
            self.min_capacity_label.grid_remove()
            self.min_capacity_entry.grid_remove()
            self.max_capacity_label.grid_remove()
            self.max_capacity_entry.grid_remove()

    def update_capacity_range(self):
        self.update_capacity_options()

    def update_decimal_places(self, event=None):
        self.update_weight_options()

    def update_capacity_decimal_places(self, event=None):
        self.update_capacity_options()

    def update_modify_edge_btn(self):
        graph_type = self.graph_type.get()
        is_w = self.additional_options['is_weighted'].get()

        if graph_type == "Red de flujo" or is_w:
            self.modify_edge_button.grid()
        else:
            self.modify_edge_button.grid_remove()
                
    def update_node_weight_see(self):
        if self.analyzer.G is None:
            messagebox.showinfo("Error", "Aún no se ha creado el grafo.")
            return

        # Crear un diálogo simple para ingresar el nodo
        label = simpledialog.askstring("Seleccionar Nodo", "Ingrese el label (#) del nodo:")

        try:
            label = int(label)
        except ValueError:
            pass

        if label is None:  # El usuario canceló el diálogo
            return

        if label not in self.analyzer.G.nodes:
            messagebox.showerror("Error", f"El nodo '{label}' no existe en el grafo.")

        # Crear subgrafo inducido
        subgraph = self.analyzer.G.subgraph([label] + list(self.analyzer.G.neighbors(label)))

        # Limpiar el frame de visualización
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Crear una nueva figura
        fig = plt.figure(figsize=(5, 4))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)

        ax = fig.add_subplot(111, position=[0, 0, 1, 1])

        # Generar el layout
        pos = nx.spring_layout(subgraph)

        is_directed = subgraph.is_directed()
        edge_counts = {}
        self_loops_count = {}

        # Dibujar aristas
        if isinstance(subgraph, (nx.MultiGraph, nx.MultiDiGraph)):
            all_edges = subgraph.edges(keys=True, data=True)
        else:
            all_edges = subgraph.edges(data=True)

        for edge in all_edges:
            if isinstance(subgraph, (nx.MultiGraph, nx.MultiDiGraph)):
                u, v, key, d = edge
            else:
                u, v, d = edge

            rad = 0
            if (u, v) in edge_counts or (v, u) in edge_counts:
                count = edge_counts.get((u, v), 0) + edge_counts.get((v, u), 0)
                count = count if is_directed else count // 2
                rad = 0.1 + 0.05 * count
                rad *= (-1) ** count
                edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
                edge_counts[(v, u)] = edge_counts.get((v, u), 0) + 1
            else:
                edge_counts[(u, v)] = edge_counts[(v, u)] = 1

            if u == v:
                if u not in self_loops_count:
                    self_loops_count[u] = 0
                self_loops_count[u] += 1

            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], ax=ax, node_size=300 if u != v else int(300 * (1 + abs(rad))),
                                connectionstyle=f"arc3,rad={rad}",
                                arrows=True)

            # Dibujar el peso/capacidad de la arista
            edge_weight = d.get('weight', '') if self.additional_options["is_weighted"].get() else ''
            edge_capacity = d.get('capacity', '') if self.graph_type.get() == "Red de flujo" else ''
            edge_label = edge_weight or edge_capacity
            if edge_label:
                if u != v:  # No es un lazo
                    edge_x = (pos[u][0] + pos[v][0]) / 2
                    edge_y = (pos[u][1] + pos[v][1]) / 2
                    edge_x += rad * (pos[v][1] - pos[u][1]) / 2
                    edge_y -= rad * (pos[v][0] - pos[u][0]) / 2
                    ax.text(edge_x, edge_y, str(edge_label), fontsize=8,
                            ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='black'))
                else:  # Es un lazo
                    offset = (self_loops_count[u] - 1) * 0.04 if self_loops_count[u] < 4 else (self_loops_count[u] - 1) * 0.02
                    edge_x = pos[u][0] + 0.03
                    edge_y = pos[u][1] + offset
                    ax.text(edge_x, edge_y, str(edge_label), fontsize=8,
                            ha='left', va='center', bbox=dict(boxstyle='round', fc='white', ec='black'))

        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8, font_weight='bold')

        ax.axis('off')

        # Crear un canvas de Matplotlib
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Añadir botones de zoom
        zoom_frame = ttk.Frame(self.plot_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT)

        # Guardar referencias
        self.canvas = canvas
        self.fig = fig
        self.ax = ax

        # Configurar eventos de zoom y pan
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.pressed = False
        self.start_pan = None

    def update_two_nodes_weight_see(self):
        if self.analyzer.G is None:
            messagebox.showinfo("Error", "Aún no se ha creado el grafo.")
            return

        class TwoNodesDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text="Nodo 1:").grid(row=0)
                tk.Label(master, text="Nodo 2:").grid(row=1)
                self.e1 = tk.Entry(master)
                self.e2 = tk.Entry(master)
                self.e1.grid(row=0, column=1)
                self.e2.grid(row=1, column=1)
                return self.e1  # initial focus

            def apply(self):
                self.node1 = self.e1.get()
                self.node2 = self.e2.get()

        dialog = TwoNodesDialog(self.master, title="Ingrese los dos nodos")

        if dialog.node1 is None or dialog.node2 is None:  # El usuario canceló el diálogo
            return

        try:
            node1 = int(dialog.node1)
            node2 = int(dialog.node2)
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese números válidos para los nodos.")
            return

        if node1 not in self.analyzer.G.nodes or node2 not in self.analyzer.G.nodes:
            messagebox.showerror("Error", f"Uno o ambos nodos no existen en el grafo.")
            return

        # Crear subgrafo inducido
        subgraph = self.analyzer.G.subgraph([node1, node2])

        # Limpiar el frame de visualización
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Crear una nueva figura
        fig = plt.figure(figsize=(5, 4))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)

        ax = fig.add_subplot(111, position=[0, 0, 1, 1])

        # Generar el layout
        pos = nx.spring_layout(subgraph)

        is_directed = subgraph.is_directed()
        edge_counts = {}
        self_loops_count = {}

        # Dibujar aristas
        if isinstance(subgraph, (nx.MultiGraph, nx.MultiDiGraph)):
            all_edges = subgraph.edges(keys=True, data=True)
        else:
            all_edges = subgraph.edges(data=True)

        for edge in all_edges:
            if isinstance(subgraph, (nx.MultiGraph, nx.MultiDiGraph)):
                u, v, key, d = edge
            else:
                u, v, d = edge

            rad = 0
            if (u, v) in edge_counts or (v, u) in edge_counts:
                count = edge_counts.get((u, v), 0) + edge_counts.get((v, u), 0)
                count = count if is_directed else count // 2
                rad = 0.1 + 0.05 * count
                rad *= (-1) ** count
                edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
                edge_counts[(v, u)] = edge_counts.get((v, u), 0) + 1
            else:
                edge_counts[(u, v)] = edge_counts[(v, u)] = 1

            if u == v:
                if u not in self_loops_count:
                    self_loops_count[u] = 0
                self_loops_count[u] += 1

            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], ax=ax, node_size=300 if u != v else int(300 * (1 + abs(rad))),
                                connectionstyle=f"arc3,rad={rad}",
                                arrows=True)

            # Dibujar el peso/capacidad de la arista
            edge_weight = d.get('weight', '') if self.additional_options["is_weighted"].get() else ''
            edge_capacity = d.get('capacity', '') if self.graph_type.get() == "Red de flujo" else ''
            edge_label = edge_weight or edge_capacity
            if edge_label:
                if u != v:  # No es un lazo
                    edge_x = (pos[u][0] + pos[v][0]) / 2
                    edge_y = (pos[u][1] + pos[v][1]) / 2
                    edge_x += rad * (pos[v][1] - pos[u][1]) / 2
                    edge_y -= rad * (pos[v][0] - pos[u][0]) / 2
                    ax.text(edge_x, edge_y, str(edge_label), fontsize=8,
                            ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='black'))
                else:  # Es un lazo
                    offset = (self_loops_count[u] - 1) * 0.04 if self_loops_count[u] < 4 else (self_loops_count[u] - 1) * 0.02
                    edge_x = pos[u][0] + 0.03
                    edge_y = pos[u][1] + offset
                    ax.text(edge_x, edge_y, str(edge_label), fontsize=8,
                            ha='left', va='center', bbox=dict(boxstyle='round', fc='white', ec='black'))

        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8, font_weight='bold')

        ax.axis('off')

        # Crear un canvas de Matplotlib
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Añadir botones de zoom
        zoom_frame = ttk.Frame(self.plot_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT)

        # Guardar referencias
        self.canvas = canvas
        self.fig = fig
        self.ax = ax

        # Configurar eventos de zoom y pan
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.pressed = False
        self.start_pan = None

#Funciones auxiliares
    def visualize_graph(self, graph, pos):
        # Limpiar el frame de visualización
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Crear una nueva figura
        fig = plt.figure(figsize=(5, 4))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)

        ax = fig.add_subplot(111, position=[0, 0, 1, 1])

        # Dibujar el grafo
        self.draw_graph(graph, pos, ax)

        ax.axis('off')

        # Crear un canvas de Matplotlib
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Añadir botones de zoom
        self.add_zoom_buttons()

        # Guardar referencias
        self.canvas = canvas
        self.fig = fig
        self.ax = ax

        # Configurar eventos de zoom y pan
        self.setup_zoom_pan_events()

    def draw_graph(self, graph, pos, ax):
        is_directed = graph.is_directed()
        edge_counts = {}
        self_loops_count = {}


        # Dibujar aristas
        if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
            all_edges = graph.edges(keys=True, data=True)
        else:
            all_edges = graph.edges(data=True)

        for edge in all_edges:
            self.draw_edge(graph, edge, pos, ax, is_directed, edge_counts, self_loops_count)

        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight='bold')

    def draw_edge(self, graph, edge, pos, ax, is_directed, edge_counts, self_loops_count):
        if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
            u, v, key, d = edge
        else:
            u, v, d = edge

        rad = self.calculate_edge_radius(u, v, is_directed, edge_counts)

        if u == v:
            self.update_self_loop_count(u, self_loops_count)

        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], ax=ax, node_size=300 if u != v else int(300 * (1 + abs(rad))),
                            connectionstyle=f"arc3,rad={rad}",
                            arrows=True) 

        self.draw_edge_label(u, v, d, pos, ax, rad, self_loops_count)
   
    def draw_edge_label(self, u, v, d, pos, ax, rad, self_loops_count):
        edge_weight = d.get('weight', '') if self.additional_options["is_weighted"].get() else ''
        edge_capacity = d.get('capacity', '') if self.graph_type.get() == "Red de flujo" else ''
        edge_label = edge_weight or edge_capacity
        
        if edge_label:
            if u != v:  # No es un lazo
                vector = pos[v] - pos[u]
                norm = np.linalg.norm(vector)
                vector /= norm
                ax.arrow(0,0,vector[0], vector[1], fc="white", ec="white", alpha=0)
                vector = np.array([vector[1], -vector[0]])
                ax.arrow(0,0,vector[0], vector[1], fc="white", ec="white", alpha=0)
                edge_x = vector[0]
                edge_y = vector[1]
                edge_x = (pos[v][0] + pos[u][0]) / 2 + edge_x * rad * norm * .5 
                edge_y = (pos[v][1] + pos[u][1]) / 2 + edge_y * rad * norm * .5

                ax.text(edge_x, edge_y, str(edge_label), fontsize=8,
                        ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='black'))
            else:  # Es un lazo
                x, y = pos[u]
                loop_count = self_loops_count.get(u, 1)
                
                # Ajustar la posición para lazos múltiples
                angle = 2 * np.pi * (loop_count - 1) / max(4, loop_count)
                r = 0.15  # Radio del lazo, ajusta según sea necesario
                
                label_x = x + r * np.cos(angle)
                label_y = y + r * np.sin(angle)
                
                ax.text(label_x, label_y, str(edge_label), fontsize=8,
                        ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='black'))

    def calculate_edge_radius(self, u, v, is_directed, edge_counts):
        key = (u, v) if u <= v else (v, u)
        if key in edge_counts:
            count = edge_counts[key]
            rad = 0.1 + 0.05 * count
            # rad *= (-1) ** count
            edge_counts[key] += 1
        else:
            rad = 0
            edge_counts[key] = 1
        return rad
    
    def update_self_loop_count(self, u, self_loops_count):
        self_loops_count[u] = self_loops_count.get(u, 0) + 1

    def add_zoom_buttons(self):
        zoom_frame = ttk.Frame(self.plot_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT)

    def setup_zoom_pan_events(self):
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.pressed = False
        self.start_pan = None
  
    def show_graph(self):
        if self.analyzer.G is None:
            messagebox.showinfo("Error", "Aún no se ha creado el grafo.")
            return

        # Obtener el grafo del analizador
        G = self.analyzer.G

        # Definir los layouts disponibles
        layouts = {
            "Resorte": nx.spring_layout,
            "Circular": nx.circular_layout,
            "Aleatorio": nx.random_layout,
            "Bonito": nx.kamada_kawai_layout
        }

        # Usar el layout seleccionado
        chosen_layout = self.layout_option.get()
        scale = self.node_distance.get()

        # Generar el layout
        if self.analyzer.pos is None:
            if scale <= 0:
                scale = 1
            self.analyzer.pos = layouts[chosen_layout](G, scale) if chosen_layout not in ["Aleatorio", "Bonito"] else layouts[chosen_layout](G)

        # Visualizar el grafo
        self.visualize_graph(G, self.analyzer.pos)

        # Actualizar los límites del gráfico después de visualizarlo
        self.update_graph_limits()

    def update_graph_limits(self):
        if hasattr(self, 'ax'):
            pos = nx.get_node_attributes(self.analyzer.G, 'pos')
            if pos:
                x_values, y_values = zip(*pos.values())
                x_margin = (max(x_values) - min(x_values)) * 0.1
                y_margin = (max(y_values) - min(y_values)) * 0.1
                self.ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
                self.ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
                self.canvas.draw()

    def zoom(self, factor):
        self.ax.set_xlim([coord * factor for coord in self.ax.get_xlim()])
        self.ax.set_ylim([coord * factor for coord in self.ax.get_ylim()])
        self.canvas.draw()

    def on_scroll(self, event):
        if event.button == 'up':
            self.zoom(1.1)
        elif event.button == 'down':
            self.zoom(0.9)

    def on_press(self, event):
        if event.button == 1:  # Left click
            self.pressed = True
            self.start_pan = (event.x, event.y)

    def on_release(self, event):
        self.pressed = False
        self.start_pan = None

    def on_motion(self, event):
        if self.pressed and self.start_pan:
            dx = event.x - self.start_pan[0]
            dy = event.y - self.start_pan[1]

            # Convertir el desplazamiento de píxeles a unidades de datos
            x_scale = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / self.fig.get_size_inches()[0] / self.fig.dpi
            y_scale = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) / self.fig.get_size_inches()[1] / self.fig.dpi

            self.ax.set_xlim(self.ax.get_xlim() - dx * x_scale)
            self.ax.set_ylim(self.ax.get_ylim() - dy * y_scale)  

            self.canvas.draw()
            self.start_pan = (event.x, event.y)

    def show_plot(self, fig=None):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        if fig is None:
            fig = plt.gcf()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    

    @staticmethod
    def generate_random_dag(num_nodes, edge_probability):
        T = nx.random_tree(n=num_nodes)
        G = nx.DiGraph(T)

        if num_nodes < 300:
            for u, v in list(G.edges()):
                if u > v:
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
            for u in G.nodes():
                for v in G.nodes():
                    if u < v and not G.has_edge(u, v) and random.random() < edge_probability:
                        G.add_edge(u, v)
        return G

    def add_node(self):
        label = simpledialog.askstring("Añadir nodo", "Ingrese el label (#) del nodo:")
        if label:
            try:
                self.analyzer.save_to_change_log()

                label = int(label)
                if label in self.analyzer.G.nodes:
                    messagebox.showinfo("Información", f"El nodo {label} ya existe en el grafo.")
                else:
                    self.analyzer.G.add_node(label)
                    self.analyzer.pos = None
                    self.show_graph()
                    messagebox.showinfo("Éxito", f"Nodo {label} añadido correctamente.")
            except ValueError:
                self.analyzer.revert_changes()
                messagebox.showerror("Error", "El label del nodo debe ser un número entero.")
   
    def add_edge(self):
        graph_type = self.graph_type.get()
        is_weighted = self.additional_options['is_weighted'].get()

        prompts = ["Ingrese el primer nodo:", "Ingrese el segundo nodo:"]
        if graph_type == "Red de flujo" or is_weighted:
            prompts.append("Ingrese el peso/capacidad de la arista:")

        results = self.multi_input_dialog("Añadir arista", prompts)

        if results:
            try:
                self.analyzer.save_to_change_log()
                node1 = int(results[0])
                node2 = int(results[1])

                # Añadir nodos si no existen
                if node1 not in self.analyzer.G.nodes:
                    self.analyzer.G.add_node(node1)
                if node2 not in self.analyzer.G.nodes:
                    self.analyzer.G.add_node(node2)

                # Determinar si es un lazo
                is_loop = node1 == node2

                # Comprobar si necesitamos cambiar el tipo de grafo
                if not isinstance(self.analyzer.G, (nx.MultiGraph, nx.MultiDiGraph)):
                    if is_loop or self.analyzer.G.has_edge(node1, node2):
                        # Cambiar a MultiGraph o MultiDiGraph
                        new_G = nx.MultiDiGraph() if self.analyzer.G.is_directed() else nx.MultiGraph()
                        new_G.add_nodes_from(self.analyzer.G.nodes(data=True))
                        new_G.add_edges_from(self.analyzer.G.edges(data=True))
                        self.analyzer.G = new_G
                        if is_loop:
                            self.graph_type.set("Pseudografo")
                        else:
                            self.graph_type.set("Multigrafo")
                elif isinstance(self.analyzer.G, nx.MultiGraph) and is_loop:
                    self.graph_type.set("Pseudografo")

                # Añadir la arista
                if graph_type == "Red de flujo" or is_weighted:
                    weight = float(results[2])
                    self.analyzer.G.add_edge(node1, node2, weight=weight)
                    self.analyzer.G.add_edge()
                    print(f"Arista añadida: {node1} - {node2}, peso/capacidad: {weight}")
                else:
                    self.analyzer.G.add_edge(node1, node2)
                    print(f"Arista añadida: {node1} - {node2}")

                self.analyzer.pos = None
                self.show_graph()
                messagebox.showinfo("Éxito", "Arista añadida correctamente.")
            except:
                self.analyzer.revert_changes()
                messagebox.showerror("Error", "Los nodos deben ser números enteros y el peso/capacidad (si aplica) debe ser un número válido.")

    def multi_input_dialog(self, title, prompts):
        class CustomDialog(simpledialog.Dialog):
            def __init__(self, parent, title, prompts):
                self.prompts = prompts
                self.entries = []
                self.results = []
                super().__init__(parent, title)

            def body(self, master):
                for i, prompt in enumerate(self.prompts):
                    tk.Label(master, text=prompt).grid(row=i, column=0, sticky="e", padx=5, pady=5)
                    entry = tk.Entry(master)
                    entry.grid(row=i, column=1, padx=5, pady=5)
                    self.entries.append(entry)
                return self.entries[0]  # initial focus

            def apply(self):
                self.results = [entry.get() for entry in self.entries]

        dialog = CustomDialog(self.master, title, prompts)
        
        if dialog.results:
            return dialog.results
        return None
   
    def remove_node(self):
        label = simpledialog.askstring("Eliminar nodo", "Ingrese el label del nodo a eliminar:")
        if label:
            if label.isnumeric():
                label = int(label)
            
            try: 
                self.analyzer.save_to_change_log()
                self.analyzer.G.remove_node(label)
                
                self.analyzer.pos = None
                self.show_graph()
            except:
                self.analyzer.revert_changes()
                messagebox.showerror('Error', f'El nodo {label} no existe.')
 
    def remove_edge(self):
        class TwoValueDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text="Ingrese el primer Nodo:").grid(row=0)
                tk.Label(master, text="Ingrese el segundo Nodo:").grid(row=1)
                tk.Label(master, text="Ingrese la cantidad (opcional):").grid(row=2)
                tk.Label(master, text="Ingrese los pesos (opcional):").grid(row=3)

                self.e1 = tk.Entry(master)
                self.e2 = tk.Entry(master)
                self.e3 = tk.Entry(master)
                self.e4 = tk.Entry(master)

                self.e1.grid(row=0, column=1)
                self.e2.grid(row=1, column=1)
                self.e3.grid(row=2, column=1)
                self.e4.grid(row=3, column=1)

                return self.e1  # inicial focus

            def apply(self):
                try:
                    self.nodo1 = self.e1.get()
                    self.nodo2 = self.e2.get()
                    self.cantidad = int(self.e3.get()) if self.e3.get().isnumeric() else 1
                    
                    # Procesar los pesos
                    peso_str = self.e4.get()
                    # Usar expresión regular para encontrar todos los números (enteros o decimales)
                    pesos = re.findall(r'-?\d+(?:\.\d+)?', peso_str)
                    # Convertir los strings a float
                    self.pesos = [float(peso) for peso in pesos]
                    
                except ValueError:
                    self.nodo1 = None
                    self.nodo2 = None
                    self.cantidad = 1
                    self.pesos = []

        dialogo = TwoValueDialog(root, title="Ingresar dos nodos")
        if dialogo.nodo1 is not None and dialogo.nodo2 is not None:
            if dialogo.nodo1.isnumeric():
                dialogo.nodo1 = int(dialogo.nodo1)
            if dialogo.nodo2.isnumeric():
                dialogo.nodo2 = int(dialogo.nodo2)
            
            try:
                self.analyzer.save_to_change_log()
                if isinstance(self.analyzer.G, (nx.MultiGraph, nx.MultiDiGraph)):
                    # Para multigrafos y pseudografos
                    edges = list(self.analyzer.G.edges(keys=True, data=True))
                    if self.analyzer.G.is_directed():
                        matching_edges = [e for e in edges if e[0] == dialogo.nodo1 and e[1] == dialogo.nodo2]
                    else:
                        matching_edges = [e for e in edges if e[0] == dialogo.nodo1 and e[1] == dialogo.nodo2 or e[1] == dialogo.nodo1 and e[0] == dialogo.nodo2]
                    
                    if matching_edges:
                        if dialogo.cantidad < len(dialogo.pesos):
                            dialogo.cantidad = len(dialogo.pesos)


                        if len(dialogo.pesos) != 0 and (self.additional_options["is_weighted"].get() or self.graph_type.get() == "Red de flujo"):
                            edge_weight = 'weight' if self.additional_options["is_weighted"].get() else ''
                            edge_capacity = 'capacity' if self.graph_type.get() == "Red de flujo" else ''
                            edge_label = edge_weight or edge_capacity
                            order_edges = deque()
                            remaining_edges = deque()
                            for i in range(len(matching_edges)):
                                if matching_edges[i][3][edge_label] in dialogo.pesos:
                                    order_edges.append(matching_edges[i])
                                else:
                                    remaining_edges.append(matching_edges[i])
                                                        
                            # Ordenar order_edges basándose en el orden de los pesos en dialogo.pesos
                            order_edges = deque(sorted(order_edges, key=lambda x: dialogo.pesos.index(x[3][edge_label])))
                            
                            order_edges.extend(remaining_edges)
                            print("Order_edges con todo", list(order_edges))
                            
                            for _ in range(min(dialogo.cantidad, len(order_edges))):
                                edge_to_remove = order_edges.popleft()
                                print("Arista a remover: ", edge_to_remove)
                                self.analyzer.G.remove_edge(edge_to_remove[0], edge_to_remove[1], edge_to_remove[2])

                        else:
                            # El código original para cuando no hay pesos especificados
                            matching_edges = deque(matching_edges)
                            for _ in range(min(dialogo.cantidad, len(matching_edges))):
                                edge_to_remove = matching_edges.popleft()
                                self.analyzer.G.remove_edge(edge_to_remove[0], edge_to_remove[1], edge_to_remove[2])

                        messagebox.showinfo('Éxito', f'Se han eliminado {min(dialogo.cantidad, len(matching_edges))} aristas entre {dialogo.nodo1} y {dialogo.nodo2}.')
                    else:
                        messagebox.showerror('Error', f'No existe la arista <{dialogo.nodo1}, {dialogo.nodo2}>.')
                else:
                    self.analyzer.G.remove_edge(dialogo.nodo1, dialogo.nodo2)
                    if dialogo.cantidad == 1:
                        messagebox.showinfo('Éxito', f'Se ha eliminado una arista entre {dialogo.nodo1} y {dialogo.nodo2}.')
                    else:
                        messagebox.showinfo('Éxito', f'Se han eliminado {dialogo.cantidad} aristas entre {dialogo.nodo1} y {dialogo.nodo2}.')
                
                # Actualizar la visualización del grafo
                self.show_graph()
            except:
                self.analyzer.revert_changes()
                messagebox.showerror('Error', f'No existe la arista <{dialogo.nodo1}, {dialogo.nodo2}>.')

    def modify_edge(self):
        class TwoValueDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text="Ingrese el primer Nodo:").grid(row=0)
                tk.Label(master, text="Ingrese el segundo Nodo:").grid(row=1)
                tk.Label(master, text="Ingrese el nuevo peso/capacidad de la arista:").grid(row=2)
                tk.Label(master, text="Ingrese el peso de la arista a modificar (caso de aristas múltiples):").grid(row=3)

                
                self.e1 = tk.Entry(master)
                self.e2 = tk.Entry(master)
                self.e3 = tk.Entry(master)
                self.e4 = tk.Entry(master)
   
                self.e1.grid(row=0, column=1)
                self.e2.grid(row=1, column=1)
                self.e3.grid(row=2, column=1)
                self.e4.grid(row=3, column=1)

                return self.e1  # inicial focus

            def apply(self):
                try:
                    self.nodo1 = int(self.e1.get()) if self.e1.get().isnumeric() else self.e1.get()
                    self.nodo2 = int(self.e2.get()) if self.e2.get().isnumeric() else self.e2.get()
                    self.new_weight = float(self.e3.get())
                    self.old_weight = float(self.e4.get())
                except ValueError:
                    self.nodo1 = None
                    self.nodo2 = None

        dialogo = TwoValueDialog(root, title="Ingresar dos nodos")

        if dialogo.nodo1 and dialogo.nodo2:
            if dialogo.new_weight is not None:
                try:
                    self.analyzer.save_to_change_log()
                    if isinstance(self.analyzer.G, (nx.MultiGraph, nx.MultiDiGraph)):
                        edges = list(self.analyzer.G.edges(keys=True, data=True))
                        print(edges)
                        matching_edges = [e for e in edges if (e[0] == dialogo.nodo1 and e[1] == dialogo.nodo2) or (e[1] == dialogo.nodo1 and e[0] == dialogo.nodo2 and not self.analyzer.G.is_directed())]
                        if matching_edges:
                            if len(matching_edges) > 1:
                                for e in matching_edges:
                                    old_weight = self.analyzer.G[e[0]][e[1]][e[2]]['weight']
                                    if np.isclose(old_weight, dialogo.old_weight):
                                        self.analyzer.G[e[0]][e[1]][e[2]]['weight'] = dialogo.new_weight
                                        messagebox.showinfo('Éxito', f'Se ha cambiado el peso de la arista <{dialogo.nodo1}, {dialogo.nodo2}> a {dialogo.new_weight}.')
                                        break
                                else:
                                    messagebox.showerror('Error', f'No existe la arista <{dialogo.nodo1}, {dialogo.nodo2}> con peso {dialogo.old_weight}.')
                                    raise Exception()

                            else:
                                old_weight = self.analyzer.G[matching_edges[0][0]][matching_edges[0][1]][matching_edges[0][2]]['weight']
                                self.analyzer.G[matching_edges[0][0]][matching_edges[0][1]][matching_edges[0][2]]['weight'] = dialogo.new_weight
                                messagebox.showinfo('Éxito', f'Se ha cambiado el peso de la arista <{dialogo.nodo1}, {dialogo.nodo2}> a {dialogo.new_weight}.')
                        else:
                            messagebox.showerror('Error', f'No existe la arista <{dialogo.nodo1}, {dialogo.nodo2}>.')
                            raise Exception()
                    else:
                        # Para grafos simples
                        old_weight = self.analyzer.G[dialogo.nodo1][dialogo.nodo2]['weight']
                        if old_weight == dialogo.old_weight:
                            self.analyzer.G[dialogo.nodo1][dialogo.nodo2]['weight'] = dialogo.new_weight
                            messagebox.showinfo('Éxito', f'Se ha cambiado el peso de la arista <{dialogo.nodo1}, {dialogo.nodo2}> a {dialogo.new_weight}.')
                        else:
                            # Buscar la arista con el peso más cercano
                            closest_edge = min([(w, e) for e, w in self.analyzer.G[dialogo.nodo1][dialogo.nodo2].items()], key=lambda x: abs(x[0] - dialogo.old_weight))
                            messagebox.showinfo('Éxito', f'Se ha cambiado el peso de la arista <{dialogo.nodo1}, {dialogo.nodo2}> a {closest_edge[0]}.')
                    
                    # Actualizar la visualización del grafo
                    self.show_graph()
                except:
                    self.analyzer.revert_changes()
                    messagebox.showerror('Error', f'No existe la arista <{dialogo.nodo1}, {dialogo.nodo2}>.')

    #Show Analisis

    def show_betweenness_centrality(self):
        betweenness = self.analyzer.betweenness_centrality()
        top_20 = dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20])

        subgraph = self.analyzer.G.subgraph(top_20.keys())

        fig, ax = plt.subplots(figsize=self.visual_size)
        pos = nx.spring_layout(subgraph)

        nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=600, node_color='lightblue')
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8, font_weight="bold")

        label_pos = {k: (v[0], v[1]+0.05) for k, v in pos.items()}
        nx.draw_networkx_labels(subgraph, label_pos, 
                                {k: f"{v:.3f}" for k, v in top_20.items()}, 
                                font_size=6, ax=ax)

        ax.set_title("Top 20 nodos por centralidad de intermediación")
        ax.axis('off')
        self.show_plot(fig)

    def analyze_information_distribution(self):
        result = self.analyzer.analyze_information_distribution()
        if isinstance(result, str):
            messagebox.showinfo("Error", result)
        else:
            fig, avg_degree = result
            self.show_plot(fig)
            messagebox.showinfo("Grado Promedio", f"Grado promedio: {avg_degree:.2f}")

    def calculate_centrality(self):
        centrality = self.analyzer.calculate_centrality()
        top_20 = dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20])
        
        subgraph = self.analyzer.G.subgraph(top_20.keys())

        fig, ax = plt.subplots(figsize=self.visual_size)
        pos = nx.spring_layout(subgraph)

        nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=800, node_color='lightblue')
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8, font_weight="bold")

        label_pos = {k: (v[0], v[1]+0.05) for k, v in pos.items()}
        nx.draw_networkx_labels(subgraph, label_pos, 
                                {k: f"{v:.3f}" for k, v in top_20.items()}, 
                                font_size=6, ax=ax)

        ax.set_title("Subgrafo con Top 20 nodos por centralidad de grado")
        ax.axis('off')
        plt.tight_layout()
        
        self.show_plot(fig)
    
    def show_longest_path(self):
        class PathDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text="Nodo de inicio:").grid(row=0)
                tk.Label(master, text="Nodo de fin:").grid(row=1)
                tk.Label(master, text="Máximo de iteraciones:").grid(row=2)
                tk.Label(master, text="Tiempo límite (segundos):").grid(row=3)
                self.e1 = tk.Entry(master)
                self.e2 = tk.Entry(master)
                self.e3 = tk.Entry(master)
                self.e4 = tk.Entry(master)
                self.e1.grid(row=0, column=1)
                self.e2.grid(row=1, column=1)
                self.e3.grid(row=2, column=1)
                self.e4.grid(row=3, column=1)
                self.e3.insert(0, "1000")  # valor por defecto
                self.e4.insert(0, "10")    # valor por defecto
                return self.e1  # initial focus

            def apply(self):
                try:
                    self.start = int(self.e1.get())
                    self.end = int(self.e2.get())
                    self.max_iterations = int(self.e3.get())
                    self.time_limit = float(self.e4.get())
                except ValueError:
                    self.start = None
                    self.end = None
                    self.max_iterations = None
                    self.time_limit = None

        dialog = PathDialog(self.master, title="Ingrese los parámetros")
        if all(v is not None for v in [dialog.start, dialog.end, dialog.max_iterations, dialog.time_limit]):
            start, end = dialog.start, dialog.end
            max_iterations, time_limit = dialog.max_iterations, dialog.time_limit
            result = self.analyzer.find_longest_path(start, end, max_iterations, time_limit)

            if isinstance(result, tuple) and len(result) == 3:
                message, longest_path, path_length = result

                if isinstance(longest_path, list) and longest_path:
                    # Visualizar el camino
                    path_edges = list(zip(longest_path, longest_path[1:]))
                    subgraph = self.analyzer.G.edge_subgraph(path_edges)

                    fig = Figure(figsize=self.visual_size, dpi=100)
                    ax = fig.add_subplot(111)
                    pos = nx.spring_layout(subgraph, k=1.5)
                    nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue', 
                            node_size=500, font_size=8, font_weight='bold')
                    edge_labels = nx.get_edge_attributes(subgraph, 'weight')
                    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=ax, font_size=6)
                    ax.set_title(f"Camino más largo de {start} a {end}", fontsize=16)
                    ax.axis('off')
                    fig.tight_layout()

                    # Limpiar el frame de visualización
                    for widget in self.plot_frame.winfo_children():
                        widget.destroy()

                    # Crear un canvas de Matplotlib
                    canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                    canvas.draw()

                    # Empaquetar el canvas directamente en plot_frame
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                    # Añadir botones de zoom
                    zoom_frame = ttk.Frame(self.plot_frame)
                    zoom_frame.pack(side=tk.BOTTOM, fill=tk.X)
                    ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT)
                    ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT)

                    # Guardar referencias
                    self.canvas = canvas
                    self.fig = fig
                    self.ax = ax

                    # Configurar eventos de zoom y pan
                    self.canvas.mpl_connect('scroll_event', self.on_scroll)
                    self.canvas.mpl_connect('button_press_event', self.on_press)
                    self.canvas.mpl_connect('button_release_event', self.on_release)
                    self.canvas.mpl_connect('motion_notify_event', self.on_motion)

                    self.pressed = False
                    self.start_pan = None

                    messagebox.showinfo("Camino más largo", message)
                else:
                    messagebox.showinfo("Camino más largo", message)
            else:
                messagebox.showinfo("Camino más largo", str(result))
        else:
            messagebox.showinfo("Error", "Entrada inválida. Por favor, ingrese números válidos para todos los campos.")

    def show_shortest_path(self):
        class PathDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text="Nodo de inicio:").grid(row=0)
                tk.Label(master, text="Nodo de fin:").grid(row=1)
                self.e1 = tk.Entry(master)
                self.e2 = tk.Entry(master)
                self.e1.grid(row=0, column=1)
                self.e2.grid(row=1, column=1)
                return self.e1  # initial focus

            def apply(self):
                self.start = int(self.e1.get())
                self.end = int(self.e2.get())

        dialog = PathDialog(self.master, title="Ingrese los nodos")
        if dialog.start is not None and dialog.end is not None:
            start, end = dialog.start, dialog.end
            shortest_path = self.analyzer.find_shortest_path(start, end)
            if isinstance(shortest_path, list):
                path_edges = list(zip(shortest_path, shortest_path[1:]))
                subgraph = self.analyzer.G.edge_subgraph(path_edges)

                fig, ax = plt.subplots(figsize=self.visual_size)
                pos = nx.spring_layout(subgraph)
                nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
                edge_labels = nx.get_edge_attributes(subgraph, 'weight')
                nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=ax)
                ax.set_title(f"Camino más corto de {start} a {end}")
                ax.axis('off')
                self.show_plot(fig)

                messagebox.showinfo("Camino más corto", f"Camino más corto de {start} a {end}: {shortest_path}")
            else:
                messagebox.showinfo("Camino más corto", shortest_path)

    def show_shortest_weighted_path(self):
        if not nx.is_weighted(self.analyzer.G):
            messagebox.showinfo("Camino de costo mínimo", "El grafo no es ponderado.")
            return
        
        class PathDialog(simpledialog.Dialog):
            def body(self, master):
                tk.Label(master, text="Nodo de inicio:").grid(row=0)
                tk.Label(master, text="Nodo de fin:").grid(row=1)
                self.e1 = tk.Entry(master)
                self.e2 = tk.Entry(master)
                self.e1.grid(row=0, column=1)
                self.e2.grid(row=1, column=1)
                return self.e1  # initial focus

            def apply(self):
                try:
                    self.start = int(self.e1.get())
                    self.end = int(self.e2.get())
                except ValueError:
                    self.start = None
                    self.end = None

        dialog = PathDialog(self.master, title="Ingrese los nodos")
        if dialog.start is not None and dialog.end is not None:
            start, end = dialog.start, dialog.end
            result = self.analyzer.shortest_weighted_path(start, end)
            if isinstance(result, tuple) and len(result) == 2:
                path, cost = result
                path_edges = list(zip(path, path[1:]))
                subgraph = self.analyzer.G.edge_subgraph(path_edges)

                fig, ax = plt.subplots(figsize=self.visual_size)
                pos = nx.spring_layout(subgraph)
                nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
                edge_labels = nx.get_edge_attributes(subgraph, 'weight')
                nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=ax)
                ax.set_title(f"Camino de costo mínimo de {start} a {end}")
                ax.axis('off')
                self.show_plot(fig)

                messagebox.showinfo("Camino ponderado más corto", 
                                    f"Camino menos costoso de {start} a {end}: {', '.join(map(str, path))}\n"
                                    f"Costo total del camino: {round(cost, 2)}")
            else:
                messagebox.showinfo("Camino de costo mínimo", str(result))
        else:
            messagebox.showinfo("Error", "Entrada inválida. Por favor, ingrese números enteros para los nodos.")
        
    def show_clustering_coefficient(self):
        local_clustering, average_clustering = self.analyzer.clustering_coefficient()

        fig, ax = plt.subplots(figsize=self.visual_size)
        ax.hist(list(local_clustering.values()), bins=50, edgecolor='black')
        ax.set_title("Distribución del coeficiente de agrupamiento local")
        ax.set_xlabel("Coeficiente de agrupamiento")
        ax.set_ylabel("Frecuencia")
        self.show_plot(fig)

        messagebox.showinfo("Coeficiente de agrupamiento", f"Coeficiente de agrupamiento promedio: {average_clustering:.4f}")

    def show_average_shortest_path(self):
        result = self.analyzer.average_shortest_path()
        if isinstance(result, float):
            messagebox.showinfo("Camino más corto promedio", f"Longitud promedio del camino más corto: {result:.4f}")
        elif result == "Grafo no generado aún":
            messagebox.showinfo("Error", result)
        elif result is None:
            messagebox.showinfo("Error", "No se puede calcular el camino más corto promedio")
        elif isinstance(result, str):
            messagebox.showinfo("Error", result)
        else:
            messagebox.showinfo("Camino más corto promedio", f"Longitud promedio del camino más corto (en componentes conexas): {result:.4f}")

root = tk.Tk()
app = NetworkAnalyzerGUI(root)
root.mainloop()


