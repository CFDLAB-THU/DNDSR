
n=20
if [[ -z $2 ]]
then
    echo ""
else
    n=$2
fi


h=""
i=slurm-$1.out
if [[ -z $3 ]]
then
    echo ""
else
    h="cat $i | $3 | "
    i=""
fi


watch --color "$h tail -n$n  $i"
#./watch2.sh "$h tail -n$n  $i"
