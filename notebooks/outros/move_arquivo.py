# Script para mover arquivos do raw_landing para os subdiretórios corretos (training/ e scoring/)

VOLUME_BASE = "/Volumes/credito_dev/bronze/raw_landing"

# Mapeamento: parte do nome do arquivo -> subdiretório de destino
FILE_ROUTING = {
    "training": "training",
    "test": "scoring",  # cs-test.csv vai para scoring
}

def mover_arquivos():
    arquivos = dbutils.fs.ls(VOLUME_BASE)

    for arquivo in arquivos:
        nome = arquivo.name

        # Ignora diretórios (já são training/, scoring/ etc.)
        if arquivo.isDir():
            continue

        destino_dir = None
        for chave, subdir in FILE_ROUTING.items():
            if chave in nome.lower():
                destino_dir = subdir
                break

        if destino_dir is None:
            print(f"⚠️  Nenhuma regra de roteamento para '{nome}', pulando.")
            continue

        origem = f"{VOLUME_BASE}/{nome}"
        destino = f"{VOLUME_BASE}/{destino_dir}/{nome}"

        print(f"Movendo {nome} -> {destino_dir}/")
        dbutils.fs.mv(origem, destino)

    print("✅ Concluído.")

mover_arquivos()