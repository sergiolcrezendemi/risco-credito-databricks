# TESTE
display(dbutils.fs.ls('/Volumes/credito_dev/silver/checkpoints/give_me_some_credit_training/'))

# se existir (mesmo vazio ou parcial), apague:
dbutils.fs.rm('/Volumes/credito_dev/silver/checkpoints/give_me_some_credit_training/', recurse=True)