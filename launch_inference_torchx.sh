

for i in $(seq 3001 3552)
do
    shard_name="sharded_${i}"
    shard_folder="sample-350BT-sharded2"
    name="BERT_beauty_toy_sport_ckt6000_inference_sharded_${i}"
    torchx run \
        --scheduler_args="fbpkg_ids=torchx_conda_mount:stable,manifold.manifoldfs:prod"  \
        fb.conda.torchrun \
        --h t16_grandteton \
        --run_as_root True \
        --env "DISABLE_NFS=1;DISABLE_OILFS=1;MANIFOLDFS_BUCKET=coin;LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so" \
        --name $name \
        -- \
        --no-python --nnodes=1 --nproc-per-node=1 \
        ./run.sh ./bert_inference.py  \
        --input_files="/mnt/mffuse/pretrain_recommendation/sample-350BT-sharded2/0.0.0/${shard_name}/" \
        --output_file='/mnt/mffuse/pretrain_recommendation/bertclassifier/predictions_beauty_toy_sport_lr_1e-5/' \
        --app_id='${app_id}' \
        --shard_num="${shard_name}" \
        --checkpoint='/mnt/mffuse/pretrain_recommendation/bertclassifier/beauty_toy_sport_lr_1e-5/checkpoint-6000' \
        # --h gtt_any \

echo $i
done
