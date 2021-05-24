# Create necessary directories
cd data || exit
mkdir -p creditratings
mkdir -p creditcard
mkdir -p mnist
mkdir -p cifar

# Move into each directory and download datasets
cd creditcard || exit
fileid="1yGyiVkcdm3YleM1HmvXqc9LQWh0qogP6"
filename="creditcard.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${filename}
tar xzf $filename
cd ..

cd mnist || exit
fileid="1Mpek9exF68x7UuOb52lXwvFU1w35UY2W"
filename="mnist.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${filename}
tar xzf $filename
cd ..

cd cifar || exit
fileid="1s0D4qnaYPie_DwCUECaKIHBIGLEg0ArT"
filename="cifar.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${filename}
tar xzf $filename
cd ..

exit
