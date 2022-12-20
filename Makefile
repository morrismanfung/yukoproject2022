# Author: Morris M. F. Chan
# Date: 2022-12-05

# General Structure of Rules.
# --------------------
# Target: Dependencies
#	Command

# Data Cleaning
bin/01-cleaned-data:
	python 02-model/01-data-cleaning.py

# Column Transformer
bin/02-column-transformer: bin/01-cleaned-data
	python 02-model/02-column-transformer.py

# KNN Training and Testing
bin/03-knn: 02-model/03-knn.py bin/01-cleaned-data bin/02-column-transformer
	python 02-model/03-knn.py

# SVC Training
bin/04-svc: 02-model/04-svc.py bin/01-cleaned-data bin/02-column-transformer
	python 02-model/04-svc.py

# SVC Testing
bin/04-svc-test: 02-model/04-svc-test.py bin/04-svc bin/01-cleaned-data bin/02-column-transformer 02-model/thresholds_used.csv
	python 02-model/04-svc-test.py

# RFC Training
bin/05-rfc: 02-model/05-rfc.py bin/01-cleaned-data bin/02-column-transformer
	python 02-model/05-rfc.py

# RFC Training
bin/05-rfc-test: 02-model/05-rfc-test.py bin/05-rfc bin/01-cleaned-data bin/02-column-transformer 02-model/thresholds_used.csv
	python 02-model/05-rfc-test.py

# NB Training
bin/06-nb: 02-model/06-nb.py bin/01-cleaned-data bin/02-column-transformer
	python 02-model/06-nb.py

# NB Testing
bin/06-nb-test: 02-model/06-nb-test.py bin/06-nb bin/01-cleaned-data bin/02-column-transformer 02-model/thresholds_used.csv
	python 02-model/06-nb-test.py

# LogReg Training
bin/07-logreg: 02-model/07-logreg.py bin/01-cleaned-data bin/02-column-transformer
	python 02-model/07-logreg.py

# LogReg Testing
bin/07-logreg-test: 02-model/07-logreg-test.py bin/07-logreg bin/01-cleaned-data bin/02-column-transformer 02-model/thresholds_used.csv
	python 02-model/07-logreg-test.py

# LSVC Training
bin/08-lsvc: 02-model/08-lsvc.py bin/01-cleaned-data bin/02-column-transformer
	python 02-model/08-lsvc.py

# LSVC Testing
bin/08-lsvc-test: 02-model/08-lsvc-test.py bin/08-lsvc bin/01-cleaned-data bin/02-column-transformer 02-model/thresholds_used.csv
	python 02-model/08-lsvc-test.py

# All Results
bin/09-results: bin/03-knn bin/04-svc-test bin/05-rfc-test bin/06-nb-test bin/07-logreg-test bin/08-lsvc-test
	python 02-model/09-results.py