

input=$(cat ${1} | tr [ { | tr ] })

echo "$input" > result.txt



