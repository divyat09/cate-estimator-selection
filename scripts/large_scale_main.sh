for ((seed=0;seed<20;seed++));
do
    sbatch scripts/large_scale.sh $seed $1
done
