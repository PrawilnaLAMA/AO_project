eval:
	python pipeline.py \
	--db-path imageDB \
	--pairs-count 10000 \
	--output-pairs pairs.txt \
	--output-csv wyniki/results.csv \
	--weights vggface2 \
	--device cuda \
	--threshold 0.8 \
	--batch-size 32 \
	--log-level INFO
