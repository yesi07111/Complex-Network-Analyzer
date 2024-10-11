import networkx
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import time
import threading
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import to_rgb

class SimulationInputWindow(tk.Toplevel):
    def __init__(self, parent, graph, pos):
        self.vectors = None
        self.graph = graph
        self.pos = pos
        self.simulation = self.simulation_iter = None
        self.simulation_max_steps = 100

        super().__init__(parent)
        self.init_widgets()
    
    def start_simulation(self):
        diff_p = (self.dif_min.get(), self.dif_max.get())
        accep_p = (self.accep_min.get(), self.accep_max.get())
        res_c = (self.res_min.get(), self.res_max.get())
        mod_c = (self.mod_min.get(), self.mod_max.get())

        self.simulation = Simulation(self.graph, diff_p, accep_p, res_c, mod_c)
        
        self.initial_info = {vec[0]: np.array(vec[1:]) for vec in self.node_data}
        
        max_norm = max(np.linalg.norm(self.initial_info[i]) for i in self.initial_info)
        
        for i in self.initial_info:
            self.initial_info[i] /= max_norm
        
        self.simulation_max_steps = self.simulation_max_steps_widget.get()
        self.simulation_iter = self.simulation.simulate_information_diffusion(self.initial_info, self.simulation_max_steps)

        self.switch_to_simulation_view()

    def init_widgets(self):
        self.title("Configuración")
        self.geometry("380x250")

        # Primer grid
        frame1 = ttk.Frame(self, padding="10")
        frame1.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame1, text="Probabilidad de difusión:").grid(row=0, column=0, sticky=tk.W)
        self.dif_min = tk.DoubleVar(value=0.0)
        dif_min = ttk.Entry(frame1, width=10, textvariable=self.dif_min)
        dif_min.grid(row=0, column=1)
        
        self.dif_max = tk.DoubleVar(value=1.0)
        dif_max = ttk.Entry(frame1, width=10, textvariable=self.dif_max)
        dif_max.grid(row=0, column=2)

        
        ttk.Label(frame1, text="Probabilidad de aceptación:").grid(row=1, column=0, sticky=tk.W)
        self.accep_min = tk.DoubleVar(value=0.0)
        accep_min = ttk.Entry(frame1, width=10, textvariable=self.accep_min)
        accep_min.grid(row=1, column=1)

        self.accep_max = tk.DoubleVar(value=1.0)
        accep_max = ttk.Entry(frame1, width=10, textvariable=self.accep_max)
        accep_max.grid(row=1, column=2)

        ttk.Label(frame1, text="Coeficiente de resiliencia:").grid(row=2, column=0, sticky=tk.W)
        self.res_min = tk.DoubleVar(value=0.0)
        res_min = ttk.Entry(frame1, width=10, textvariable=self.res_min)
        res_min.grid(row=2, column=1)

        self.res_max = tk.DoubleVar(value=1.0)
        res_max = ttk.Entry(frame1, width=10, textvariable=self.res_max)
        res_max.grid(row=2, column=2)

        ttk.Label(frame1, text="Coef. modificación de información:").grid(row=3, column=0, sticky=tk.W)
        self.mod_min = tk.DoubleVar(value=0.0)
        mod_min = ttk.Entry(frame1, width=10, textvariable=self.mod_min)
        mod_min.grid(row=3, column=1)

        self.mod_max = tk.DoubleVar(value=1.0)
        mod_max = ttk.Entry(frame1, width=10, textvariable=self.mod_max)
        mod_max.grid(row=3, column=2)

        # Segundo grid
        frame2 = ttk.Frame(self, padding="10")
        frame2.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame2, text="Cantidad de nodos a difundir:").grid(row=0, column=0, sticky=tk.W)
        self.cantidad_nodos = ttk.Entry(frame2, width=10)
        self.cantidad_nodos.grid(row=0, column=1)

        ttk.Button(frame2, text="Continuar", command=self.open_node_window).grid(row=0, column=2)

        # Tercer grid
        frame3 = ttk.Frame(self, padding="10")
        frame3.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame3, text="Cantidad de pasos a simular:").grid(row=0, column=0, sticky=tk.W)
        self.simulation_max_steps_widget = tk.IntVar(value=100)
        simulation_max_steps_widget = ttk.Entry(frame3, width=10, textvariable=self.simulation_max_steps_widget)
        simulation_max_steps_widget.grid(row=0, column=1)

        # Botones finales
        ttk.Button(self, text="Continuar", command=self.start_simulation).grid(row=3, column=0, sticky=tk.E, padx=5, pady=5)
        ttk.Button(self, text="Cancelar", command=self.destroy).grid(row=3, column=0, sticky=tk.E, padx=(0,80), pady=5)
    
    def switch_to_simulation_view(self):
        self.title("Simulation Input")
        self.geometry("800x600")
        
        # Limpiar la ventana actual
        for widget in self.winfo_children():
            widget.destroy()
        
        # Crear el canvas para la imagen de matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=3, padx=0, pady=0, sticky="nsew")
        
        # Crear los botones de control
        self.prev_button = ttk.Button(self, text="Prev", command=self.prev_image)
        self.prev_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.play_pause_button = ttk.Button(self, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.grid(row=1, column=1, padx=5, pady=5)
        
        self.next_button = ttk.Button(self, text="Next", command=self.next_image)
        self.next_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Configurar el grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_zoom_pan_events()
        
        # Mostrar la primera imagen
        self.images = [self.initial_info]
        self.current_image = 0
        self.is_playing = False
        self.update_image(self.images[self.current_image])

    def update_image(self, step):
        # Limpiar el axes actual
        self.ax.clear()
        self.ax.axis('off')
    
        color = [self.array_to_rgb_hex(step.get(i, None)) for i in sorted(self.graph.nodes())]
        
        if len(self.graph.edges) < 1000:
            nx.draw_networkx_edges(self.graph, self.pos, alpha=0.2, width=0.5, ax=self.ax)

        nx.draw_networkx_nodes(self.graph, self.pos, node_size=300, node_color=color, ax=self.ax, edgecolors="black")
        nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax, font_size=8, font_weight='bold')
        
        self.canvas.draw()
    
    def prev_image(self):
        if self.current_image > 0:
            self.current_image -= 1
            self.update_image(self.images[self.current_image])
        else:
            tk.messagebox.showerror("Error", "No es posible retroceder más en la simulación.")
    
    def next_image(self):
        if self.current_image < self.simulation_max_steps:
            self.current_image += 1
            while self.current_image >= len(self.images):
                self.images.append(next(self.simulation_iter))
            self.update_image(self.images[self.current_image])
        else:
            tk.messagebox.showerror("Error", "No es posible avanzar más en la simulación.")
    
    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_button.config(text="Pause")
            self.play_thread = threading.Thread(target=self.play_animation)
            self.play_thread.start()
        else:
            if self.play_thread:
                self.play_thread.join(timeout=2)
                if self.play_thread.is_alive():
                    print("Advertencia: El hilo no se detuvo limpiamente")
            self.play_pause_button.config(text="Play")
    
    def play_animation(self):
        try:
            while self.is_playing:
                time.sleep(0.5)
                self.next_image()
        except:
            pass

    def open_node_window(self):
        try:
            n = int(self.cantidad_nodos.get())
            n = min(n, len(self.graph.edges))
            self.node_window = NodeInfoInputWindow(self.master, n)
            self.wait_window(self.node_window)
            if hasattr(self.node_window, 'result'):
                self.node_data = self.node_window.result
            else:
                raise Exception('something went wrong')
        except ValueError:
            tk.messagebox.showerror("Error", "Por favor, introduce un número entero válido.")

    def setup_zoom_pan_events(self):
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.pressed = False
        self.start_pan = None
    
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

    @staticmethod
    def array_to_rgb_hex(arr):
        if arr is None:
            return 'white'
        
        arr = np.asarray(arr)
        if arr.shape[0] != 3:
            raise ValueError("El array debe tener 3 elementos (R, G, B)")
        
        if not np.isclose(np.linalg.norm(arr), 1, atol=1e-6):
            raise ValueError("La norma del array debe ser 1")
        
        rgb = np.clip(arr * 255, 0, 255).astype(int)
        
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        
        return hex_color

class NodeInfoInputWindow(tk.Toplevel):
    def __init__(self, parent, n):
        super().__init__(parent)
        self.title("Introducir Nodos")
        self.geometry("300x400")

        self.vectors = []

        # Crear un Canvas
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Crear una Scrollbar y asociarla con el Canvas
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Crear un frame dentro del Canvas
        self.color_options = [
            "red", "green", "blue", "yellow", "purple", "orange", 
            "pink", "cyan", "magenta", "brown", "gray", "black"
        ]

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor='nw')

        for i in range(n):
            frame = ttk.Frame(self.frame, padding="5")
            frame.pack(fill=tk.X)
            
            ttk.Label(frame, text=f"Nodo {i+1}:").pack(side=tk.LEFT)
            
            node = ttk.Entry(frame, width=5)
            node.pack(side=tk.LEFT)
            
            color_var = tk.StringVar(frame)
            color_var.set(self.color_options[0])  # valor por defecto
            color_dropdown = ttk.Combobox(frame, textvariable=color_var, values=self.color_options, width=10)
            color_dropdown.pack(side=tk.LEFT)
            
            self.vectors.append((node, color_var))

        ttk.Button(self.frame, text="Aceptar", command=self.store_vectors).pack(pady=10)

        # Actualizar el scrollregion después de agregar todos los widgets
        self.frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Vincular la rueda del ratón al scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def store_vectors(self):
        try:
            vec = []
            for node, color_var in self.vectors:
                node_num = int(node.get())
                color_name = color_var.get()
                rgb_vector = (node_num,) + to_rgb(color_name)
                vec.append(rgb_vector)
            
            self.result = vec
            self.canvas.unbind_all("<MouseWheel>")
            self.destroy()
        except:
            tk.messagebox.showerror("Error", "Alguno/s de los datos introducidos no son validos")
    
class Simulation:
    def __init__(self, graph, difussion_p, acceptance_p, resilience_c, modif_c):
        self.graph = graph

        self.difussion_prob_dist = lambda n: np.random.uniform(*difussion_p, n)
        self.acceptance_prob_dist = lambda n: np.random.uniform(*acceptance_p, n)
        self.resilience_coef_dist = lambda n: np.random.uniform(*resilience_c, n)
        self.info_mod_coef_dist = lambda n: np.random.uniform(*modif_c, n)

        self.add_node_properties()
    
    def add_node_properties(self):
        n = self.graph.number_of_nodes()

        difussion_probs = self.difussion_prob_dist(n)
        acceptance_probs = self.acceptance_prob_dist(n)
        resilience_coefs = self.resilience_coef_dist(n)
        info_mod_coefs = self.info_mod_coef_dist(n)

        for i, node in enumerate(self.graph.nodes()):
            self.graph.nodes[node]['diffusion_probability'] = difussion_probs[i]
            self.graph.nodes[node]['acceptance_probability'] = acceptance_probs[i]
            self.graph.nodes[node]['resilience_coefficient'] = resilience_coefs[i]
            self.graph.nodes[node]['information_modification_coefficient'] = info_mod_coefs[i]
    
    def simulate_information_diffusion(self, initial_info, max_steps=100, similarity_threshold=0.9):
        """
        Simula la difusión de información en un grafo.

        param G: Grafo de NetworkX con las propiedades requeridas en cada nodo
        param initial_info: Dict con nodos iniciales y su información (arrays numpy)
        param max_steps: Número máximo de pasos de simulación
        param similarity_threshold: Umbral de similitud para considerar información como "similar"
        
        yield: Estado actual de la información en la red en cada paso
        """
        # Inicializar el estado de la información
        info_state = defaultdict(lambda: None)
        info_state.update(initial_info)

        sorted_nodes = sorted(self.graph.nodes())

        for step in range(max_steps):
            # Copiar el estado actual para actualizarlo
            new_info_state = info_state.copy()

            # Para cada nodo en el grafo
            for node in sorted_nodes:
                neighbors_info = []
                
                for neighbor in self.graph.neighbors(node):
                    if info_state[neighbor] is not None and self.graph.nodes[neighbor]['diffusion_probability'] > np.random.random():
                        neighbors_info.append(info_state[neighbor])
                
                if len(neighbors_info) == 0:
                    continue

                info, count = None, -1
                
                for _info in neighbors_info:
                    _count = sum(1 for neighbor_info in neighbors_info if self.is_similar(_info, neighbor_info, similarity_threshold))

                    if _count > count:
                        info, count = _info, _count
                
                # Tomar la info que sea similar a mas vecinos
                info = self.modify_information(info, self.graph.nodes[node]['information_modification_coefficient'])
                
                # Entre mas vecinos tengan info similar mas probable es que la acepte
                acceptance_prob = self.graph.nodes[node]['acceptance_probability'] ** (len(list(self.graph.neighbors(node))) + 1 - count)
                
                if info_state[node] is None and acceptance_prob < np.random.random(): # El nodo no tiene informacion
                    new_info_state[node] = info
                
                else: # El nodo ya tiene informacion
                    acceptance_prob /= self.graph.nodes[neighbor]['resilience_coefficient']
                    if np.random.random() > acceptance_prob:
                        new_info_state[neighbor] = info
            # Actualizar el estado de la información
            info_state = new_info_state
            yield info_state

    @staticmethod
    def modify_information(info, modification_coef):
        """Modifica la información basándose en el coeficiente de modificación."""
        noise = np.random.normal(0, modification_coef, size=info.shape)
        modified = info + noise
        return modified / np.linalg.norm(modified)  # Normalizar para mantener norma 1

    @staticmethod
    def is_similar(info1, info2, threshold):
        """Comprueba si dos informaciones son similares basándose en su producto escalar."""
        return np.dot(info1, info2) > threshold

if __name__ == '__main__':
    root = tk.Tk()
    button = ttk.Button(root, text="Abrir ventana emergente", command=lambda: SimulationInputWindow(root))
    button.pack(padx=20, pady=20)
    root.mainloop()