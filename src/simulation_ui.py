import tkinter as tk
from tkinter import ttk

class VentanaEmergente(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Configuración")
        self.geometry("500x400")

        # Primer grid
        frame1 = ttk.Frame(self, padding="10")
        frame1.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame1, text="Probabilidad de difusión:").grid(row=0, column=0, sticky=tk.W)
        self.dif_min = ttk.Entry(frame1, width=10)
        self.dif_min.grid(row=0, column=1)
        self.dif_max = ttk.Entry(frame1, width=10)
        self.dif_max.grid(row=0, column=2)

        ttk.Label(frame1, text="Probabilidad de aceptación:").grid(row=1, column=0, sticky=tk.W)
        self.acep_min = ttk.Entry(frame1, width=10)
        self.acep_min.grid(row=1, column=1)
        self.acep_max = ttk.Entry(frame1, width=10)
        self.acep_max.grid(row=1, column=2)

        ttk.Label(frame1, text="Coeficiente de resiliencia:").grid(row=2, column=0, sticky=tk.W)
        self.res_min = ttk.Entry(frame1, width=10)
        self.res_min.grid(row=2, column=1)
        self.res_max = ttk.Entry(frame1, width=10)
        self.res_max.grid(row=2, column=2)

        ttk.Label(frame1, text="Coef. modificación de información:").grid(row=3, column=0, sticky=tk.W)
        self.mod_min = ttk.Entry(frame1, width=10)
        self.mod_min.grid(row=3, column=1)
        self.mod_max = ttk.Entry(frame1, width=10)
        self.mod_max.grid(row=3, column=2)

        # Segundo grid
        frame2 = ttk.Frame(self, padding="10")
        frame2.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame2, text="Cantidad de nodos a difundir:").grid(row=0, column=0, sticky=tk.W)
        self.cantidad_nodos = ttk.Entry(frame2, width=10)
        self.cantidad_nodos.grid(row=0, column=1)

        ttk.Button(frame2, text="Continuar", command=self.abrir_ventana_nodos).grid(row=0, column=2)

        # Botones finales
        ttk.Button(self, text="Continuar", command=self.guardar_datos).grid(row=2, column=0, sticky=tk.E, padx=5, pady=5)
        ttk.Button(self, text="Cancelar", command=self.destroy).grid(row=2, column=0, sticky=tk.E, padx=(0,80), pady=5)

    def abrir_ventana_nodos(self):
        try:
            n = int(self.cantidad_nodos.get())
            VentanaNodos(self, n)
        except ValueError:
            tk.messagebox.showerror("Error", "Por favor, introduce un número entero válido.")

    def guardar_datos(self):
        # Aquí puedes agregar la lógica para guardar los datos
        print("Datos guardados")
        self.destroy()

class VentanaNodos(tk.Toplevel):
    def __init__(self, parent, n):
        super().__init__(parent)
        self.title("Introducir Nodos")
        self.geometry("300x400")

        self.vectores = []

        for i in range(n):
            frame = ttk.Frame(self, padding="5")
            frame.pack(fill=tk.X)
            ttk.Label(frame, text=f"Nodo {i+1}:").pack(side=tk.LEFT)
            v1 = ttk.Entry(frame, width=5)
            v1.pack(side=tk.LEFT)
            v2 = ttk.Entry(frame, width=5)
            v2.pack(side=tk.LEFT)
            v3 = ttk.Entry(frame, width=5)
            v3.pack(side=tk.LEFT)
            self.vectores.append((v1, v2, v3))

        ttk.Button(self, text="Aceptar", command=self.guardar_vectores).pack(pady=10)

    def guardar_vectores(self):
        # Aquí puedes agregar la lógica para guardar los vectores
        print("Vectores guardados")
        self.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    button = ttk.Button(root, text="Abrir ventana emergente", command=lambda: VentanaEmergente(root))
    button.pack(padx=20, pady=20)
    root.mainloop()