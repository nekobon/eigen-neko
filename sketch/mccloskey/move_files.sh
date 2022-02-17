# To move my images to the notebooks folder

echo `pwd`
from_path="output/$1"
to_path="notebooks/images/$1"

if [[ ! -e "$from_path" ]]; then
    echo "Could not find $from_path"
fi

if [[ -e "$to_path" ]]; then
    echo "rm -r $to_path"
    rm -r "$to_path"
fi

echo "cp -r $from_path $to_path"
cp -r "$from_path" "$to_path"

echo "echo $from_path > $to_path/info.txt"
echo "$from_path" > "$to_path/info.txt"
