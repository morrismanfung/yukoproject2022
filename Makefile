# Target: Dependencies
#	Code

01-data/test.csv:
	python 02-model/dummy4make.py

02-model/01-saved-model/01-pipe_knn_opt.joblib 02-model/02-saved-scores/01-knn_dict.joblib: 02-model/03-knn.py 01-data/train.csv 01-data/test.csv
	python 02-model/03-knn.py

02-model/01-saved-model/02-pipe_svc_opt.joblib 02-model/02-saved-scores/02-svc_dict_tmp.joblib 02-model/02-saved-scores/02-svc-pr-purve.png: \
	02-model/04-svc.py 01-data/train.csv
	python 02-model/04-svc.py

02-model/02-saved-scores/02-svc_dict.joblib: \
	02-model/04-svc-test.py 02-model/01-saved-model/02-pipe_svc_opt.joblib 01-data/train.csv 01-data/test.csv
	python 02-model/04-svc-test.py

02-model/01-saved-model/03-pipe_rfc_opt.joblib 02-model/02-saved-scores/03-rfc_dict_tmp.joblib 02-model/02-saved-scores/03-rfc-pr-purve.png: \
	02-model/05-rfc.py 01-data/train.csv
	python 02-model/05-rfc.py

02-model/02-saved-scores/03-rfc_dict.joblib: \
	02-model/05-rfc-test.py 02-model/01-saved-model/03-pipe_rfc_opt.joblib 01-data/train.csv 01-data/test.csv
	python 02-model/05-rfc-test.py

02-model/01-saved-model/04-pipe_nb.joblib 02-model/02-saved-scores/04-nb_dict_tmp.joblib 02-model/02-saved-scores/04-nb-pr-purve.png: \
	02-model/06-nb.py 01-data/train.csv
	python 02-model/06-nb.py

02-model/02-saved-scores/04-nb_dict.joblib: \
	02-model/06-nb-test.py 02-model/01-saved-model/04-pipe_nb.joblib 01-data/train.csv 01-data/test.csv
	python 02-model/06-nb-test.py

02-model/01-saved-model/05-pipe_logreg_opt.joblib 02-model/02-saved-scores/05-logreg_dict_tmp.joblib 02-model/02-saved-scores/05-logreg-pr-purve.png: \
	02-model/07-logreg.py 01-data/train.csv
	python 02-model/07-logreg.py

02-model/02-saved-scores/05-logreg_dict.joblib: \
	02-model/07-logreg-test.py 02-model/01-saved-model/05-pipe_logreg_opt.joblib 01-data/train.csv 01-data/test.csv
	python 02-model/07-logreg-test.py

02-model/01-saved-model/06-pipe_lsvc_opt.joblib 02-model/02-saved-scores/06-lsvc_dict_tmp.joblib 02-model/02-saved-scores/06-lsvc-pr-purve.png: \
	02-model/08-lsvc.py 01-data/train.csv
	python 02-model/08-lsvc.py

02-model/02-saved-scores/06-lsvc_dict.joblib: \
	02-model/08-lsvc-test.py 02-model/01-saved-model/06-pipe_lsvc_opt.joblib 01-data/train.csv 01-data/test.csv
	python 02-model/08-lsvc-test.py