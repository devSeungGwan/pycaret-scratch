from pycaret.datasets import get_data
from pycaret.clustering import setup, create_model, assign_model

jewellery = get_data('jewellery')
clu = setup(data=jewellery)
kmeans = create_model('kmeans')
kmeans_results = assign_model(kmeans)