import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, Counter, defaultdict

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

    def generate_graph(self, graph_type, is_directed, is_weighted, is_acyclic, num_nodes, edge_probability):
        if is_acyclic:
            if is_directed:
                G = self.generate_random_dag(num_nodes, edge_probability)
            else:
                G = nx.random_tree(num_nodes)
            
            if graph_type == "Multigrafo":
                G = nx.MultiDiGraph(G) if is_directed else nx.MultiGraph(G)

                if edge_probability > 0.90:
                    edge_probability = 0.90
                elif edge_probability < 0:
                    edge_probability = 0

                for u, v in list(G.edges()):
                    while random.random() < edge_probability:
                        G.add_edge(u, v)

            elif graph_type == "Pseudografo":
                G = nx.MultiDiGraph(G) if is_directed else nx.MultiGraph(G)

                if edge_probability > 0.90:
                    edge_probability = 0.90
                elif edge_probability < 0:
                    edge_probability = 0

                for i in list(G.nodes()):
                    while random.random() < edge_probability:
                        G.add_edge(i, i)
                for u, v in list(G.edges()):
                    while random.random() < edge_probability:
                        G.add_edge(u, v)
        

        elif graph_type == "Multigrafo":
            if is_directed:
                G = nx.MultiDiGraph()
            else:
                G = nx.MultiGraph()
            G.add_nodes_from(range(num_nodes))

            if edge_probability > 0.90:
                edge_probability = 0.90
            elif edge_probability < 0:
                edge_probability = 0

            for i in range(num_nodes):
                for j in range(num_nodes):
                    while random.random() < edge_probability and i != j:
                        G.add_edge(i, j)
        elif graph_type == "Pseudografo":
            if is_directed:
                G = nx.MultiDiGraph()
            else:
                G = nx.MultiGraph()
            G.add_nodes_from(range(num_nodes))

            if edge_probability > 0.90:
                edge_probability = 0.90
            elif edge_probability < 0:
                edge_probability = 0

            for i in range(num_nodes):
                for j in range(i, num_nodes):  # Incluye i==j para lazos
                    while random.random() < edge_probability:
                        G.add_edge(i, j)

        elif graph_type == "Red de flujo":
            G = nx.DiGraph()
            
            G.add_nodes_from(range(num_nodes))
    
            origin = 0
            end = num_nodes - 1
    
            # Crear un camino inicial del origen al receptor
            path = list(range(num_nodes))
            random.shuffle(path[1:-1])  # Mezclar los nodos intermedios
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i+1])
    
            # Añadir aristas adicionales con cierta probabilidad
            for i in range(1, num_nodes - 1):  # Excluimos el origen y el receptor
                for j in range(1, num_nodes - 1):  # Excluimos el origen y el receptor
                    if i != j and random.random() < edge_probability:
                        G.add_edge(i, j)
    
            # Asegurar que cada nodo tenga al menos una conexión hacia adelante y hacia atrás
            for node in range(1, num_nodes - 1):
                if G.out_degree(node) == 0:
                    target = random.choice(range(node + 1, num_nodes))
                    G.add_edge(node, target)
                if G.in_degree(node) == 0:
                    source = random.choice(range(0, node))
                    G.add_edge(source, node)
    
            if not self.additional_options["weight_range"].get() and self.capacity_type_menu.get() == "Racional":
                min_capacity = 0
                max_capacity = 1
            else:
                min_capacity = self.min_capacity.get()
                max_capacity = self.max_capacity.get()

            for (u, v) in G.edges():
                if self.capacity_type_menu.get() == "Entero":
                    G[u][v]['capacity'] = random.randint(min_capacity, max_capacity)
                else:
                    roundvalue = self.additional_options["decimal_places"].get()
                    G[u][v]['capacity'] = round(random.uniform(min_capacity, max_capacity), roundvalue)

    
            G.nodes[origin]['label'] = 'Origin'
            G.nodes[end]['label'] = 'End'
    
            for pred in list(G.predecessors(origin)):
                G.remove_edge(pred, origin)
            for succ in list(G.successors(end)):
                G.remove_edge(end, succ)
            
        else:
            max_edges = (num_nodes * (num_nodes - 1)) * (1 if is_directed else .5)
            expected_edges = np.floor(max_edges * edge_probability)

            num_edges = max(0, min(expected_edges, max_edges))

            G = nx.gnm_random_graph(n=num_nodes, m=num_edges, directed=is_directed)        
    
        if is_weighted and graph_type != "Red de flujo":
            min_weight = self.min_weight.get()
            max_weight = self.max_weight.get()
            
            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                all_edges = G.edges(keys=True)
            else:
                all_edges = G.edges()

            for edge in all_edges:
                if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                    u, v, key = edge
                else:
                    u, v = edge

                if self.weight_type_menu.get() != "Entero":
                    roundvalue = self.additional_options["decimal_places"].get()
                    weight = round(random.uniform(min_weight, max_weight), roundvalue)
                else:
                    if not self.additional_options["weight_range"].get():
                        min_weight = 5
                        max_weight = 30
                    else:
                        min_weight = round(self.min_weight.get(), 0)
                        max_weight = round(self.max_weight.get(), 0) 
                    print(f"Pesos en caso de entero: {min_weight}, {max_weight}")
                    weight = random.randint(min_weight, max_weight)

                if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                    G[u][v][key]['weight'] = weight
                else:
                    G[u][v]['weight'] = weight
            self.node_weight_see.grid(row=17, column=0)
            self.two_nodes_weight_see.grid(row=17, column=1)

        self.G = G
    
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
