import os
import torch

def configurar_cuda_visible(tarjetas_seleccionadas):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cuda_visible = ",".join(map(str, tarjetas_seleccionadas))
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

def listar_gpus_disponibles():
    if torch.cuda.is_available():
        cantidad_gpus = torch.cuda.device_count()
        print(f"Se encontraron {cantidad_gpus} GPUs disponibles:")
        for i in range(cantidad_gpus):
            nombre_gpu = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {nombre_gpu}")
    else:
        print("No se encontraron GPUs disponibles.")

    
if __name__ == "__main__":

    @configurar_cuda_visible([1,2])
    def entrenar_en_gpu():
        import torch
        device = torch.cuda.current_device()
        print(f"Entrenamiento en GPU: {device} - {torch.cuda.get_device_name(device)}")
        
    entrenar_en_gpu()
    listar_gpus_disponibles()