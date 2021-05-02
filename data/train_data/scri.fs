ls $1 |head -n 25 > s1.txt

while read file; do
git add "$1$file"
done < s1.txt
rm s1.txt
