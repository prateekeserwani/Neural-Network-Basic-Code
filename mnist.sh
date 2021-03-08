DIR="./raw"
if [ -d "$DIR" ]; then
	echo "Download directory already exists"
else
	mkdir $DIR
fi
cd raw
url=http://yann.lecun.com/exdb/mnist/
declare -a StringArray=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz" )

for FILE in ${StringArray[@]}; do
	if test -f "$FILE"; then
		echo "$FILE already exists"
	else	
		echo "not exist"
		echo ${url}${FILE}
		wget "${url}${FILE}"
		gzip -d $FILE
	fi
done 
cd ..
python create_dataset.py
