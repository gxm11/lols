.PHONY: clean log submit update

update:
	git pull

submit:
	sbatch lib/config/submit.sh

clean:
	rm -rf work/* lols.db lols.log

log:
	tail -f lols.log