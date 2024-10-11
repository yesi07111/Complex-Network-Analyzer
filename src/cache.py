import networkx
class Cache:
    def __init__(self):
        self.cache_data = {}
    def has_cache(self, function_name, G):
        if function_name not in self.cache_data:
            return False
        
        cached_data = self.cache_data[function_name]['graph_data']
        current_data = self._get_graph_data(G)
        
        return cached_data == current_data
    def get_cache(self, function_name):
        return self.cache_data[function_name]['figure']
    # def save_to_cache(self, function_name, G, fig):
    #     self.cache_data[function_name] = {
    #         'graph_data': self._get_graph_data(G),
    #         'figure': fig
    #     }
    def save_to_cache(self, function_name, G, fig):
        self.cache_data[function_name] = {
            'graph_data': self._get_graph_data(G),
            'figure': fig
        }
    def _get_graph_data(self, G):
        return {
            'nodes': set(G.nodes()),
            'edges': set(G.edges()),
            'weights': {(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}
        }