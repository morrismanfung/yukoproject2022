# Author: Morris M. F. Chan
# Date: 2022-12-05

# General Structure of Rules.
# --------------------
# Target: Dependencies
#	Command

# For testing only
01-data/test.csv:
	python 02-model/dummy4make.py

# KNN Training and Testing
02-model/01-saved-model/01-pipe_knn_opt.joblib 02-model/02-saved-scores/01-knn_dict.pkl: \
# Dependencies
	02-model/03-knn.py 01-data/train.csv 01-data/test.csv
# Command to Run
	python 02-model/03-knn.py

# SVC Training
02-model/01-saved-model/02-pipe_svc_opt.joblib 02-model/02-saved-scores/02-svc_dict_tmp.pkl 02-model/02-saved-scores/02-svc-pr-purve.png: \
# Dependencies
	02-model/04-svc.py 01-data/train.csv
# Command to Run
	python 02-model/04-svc.py

# SVC Testing
02-model/02-saved-scores/02-svc_dict.pkl: \
# Dependencies
	02-model/04-svc-test.py 02-model/01-saved-model/02-pipe_svc_opt.joblib 01-data/train.csv 01-data/test.csv 02-model/thresholds_used.csv
# Command to Run
	python 02-model/04-svc-test.py

# RFC Training
02-model/01-saved-model/03-pipe_rfc_opt.joblib 02-model/02-saved-scores/03-rfc_dict_tmp.pkl 02-model/02-saved-scores/03-rfc-pr-purve.png: \
# Dependencies
	02-model/05-rfc.py 01-data/train.csv
# Command to Run
	python 02-model/05-rfc.py

# RFC Training
02-model/02-saved-scores/03-rfc_dict.pkl: \
# Dependencies
	02-model/05-rfc-test.py 02-model/01-saved-model/03-pipe_rfc_opt.joblib 01-data/train.csv 01-data/test.csv 02-model/thresholds_used.csv
# Command to Run
	python 02-model/05-rfc-test.py

# NB Training
02-model/01-saved-model/04-pipe_nb.joblib 02-model/02-saved-scores/04-nb_dict_tmp.pkl 02-model/02-saved-scores/04-nb-pr-purve.png: \
# Dependencies
	02-model/06-nb.py 01-data/train.csv
# Command to Run
	python 02-model/06-nb.py

# NB Testing
02-model/02-saved-scores/04-nb_dict.pkl: \
# Dependencies
	02-model/06-nb-test.py 02-model/01-saved-model/04-pipe_nb.joblib 01-data/train.csv 01-data/test.csv 02-model/thresholds_used.csv
# Command to Run
	python 02-model/06-nb-test.py

# LogReg Training
02-model/01-saved-model/05-pipe_logreg_opt.joblib 02-model/02-saved-scores/05-logreg_dict_tmp.pkl 02-model/02-saved-scores/05-logreg-pr-purve.png: \
# Dependencies
	02-model/07-logreg.py 01-data/train.csv
# Command to Run
	python 02-model/07-logreg.py

# LogReg Testing
02-model/02-saved-scores/05-logreg_dict.pkl: \
# Dependencies
	02-model/07-logreg-test.py 02-model/01-saved-model/05-pipe_logreg_opt.joblib 01-data/train.csv 01-data/test.csv 02-model/thresholds_used.csv
# Command to Run
	python 02-model/07-logreg-test.py

# LSVC Training
02-model/01-saved-model/06-pipe_lsvc_opt.joblib 02-model/02-saved-scores/06-lsvc_dict_tmp.pkl 02-model/02-saved-scores/06-lsvc-pr-purve.png: \
# Dependencies
	02-model/08-lsvc.py 01-data/train.csv
# Command to Run
	python 02-model/08-lsvc.py

# LSVC Testing
02-model/02-saved-scores/06-lsvc_dict.pkl: \
	02-model/08-lsvc-test.py 02-model/01-saved-model/06-pipe_lsvc_opt.joblib 01-data/train.csv 01-data/test.csv 02-model/thresholds_used.csv
	python 02-model/08-lsvc-test.py

# All Results
02-model/02-saved-scores/07-cross-valiidation-results.csv 02-model/02-saved-scores/08-test-results.csv: \
	02-model/09-results.py 02-model/02-saved-scores/01-knn_dict.pkl 02-model/02-saved-scores/02-svc_dict.pkl \
	02-model/02-saved-scores/03-rfc_dict.pkl 02-model/02-saved-scores/04-nb_dict.pkl \
	02-model/02-saved-scores/05-logreg_dict.pkl 02-model/02-saved-scores/06-lsvc_dict.pkl
	python 02-model/09-results.py