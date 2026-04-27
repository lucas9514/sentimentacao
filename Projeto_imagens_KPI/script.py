from PIL import Image
from rembg import remove

# Caminhos
input_path = "input.png"
arco_path = "arco.png"
output_path = "output.png"

# Abrir imagem
input_image = Image.open(input_path)

# Remover fundo
no_bg = remove(input_image).convert("RGBA")

# Criar base quadrada da foto
size = max(no_bg.size)
square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
square.paste(no_bg, ((size - no_bg.width) // 2, (size - no_bg.height) // 2), no_bg)

# Canvas final
final_size = 500
canvas = Image.new("RGBA", (final_size, final_size), (0, 0, 0, 0))

# Foto maior
photo_size = 510
photo = square.resize((photo_size, photo_size))
photo_x = (final_size - photo_size) // 2
photo_y = (final_size - photo_size) // 2
canvas.paste(photo, (photo_x, photo_y), photo)

# Abrir arco
arco_original = Image.open(arco_path).convert("RGBA")

# Corrigir proporção do arco colocando em base quadrada
arco_base_size = max(arco_original.width, arco_original.height)
arco_square = Image.new("RGBA", (arco_base_size, arco_base_size), (0, 0, 0, 0))

arco_x_base = (arco_base_size - arco_original.width) // 2
arco_y_base = (arco_base_size - arco_original.height) // 2

arco_square.paste(arco_original, (arco_x_base, arco_y_base), arco_original)

# Redimensionar arco já corrigido
arco_size = 870
arco = arco_square.resize((arco_size, arco_size))

# Colar arco no centro
arco_x = (final_size - arco_size) // 2 - 1
arco_y = (final_size - arco_size) // 2 + 21
canvas.paste(arco, (arco_x, arco_y), arco)

# Salvar
canvas.save(output_path)
print("Imagem criada com sucesso!")