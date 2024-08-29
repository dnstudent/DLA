from src.models.mcdropout import MCSampler

def plot_mc_performance(model, day_data, sample_size, ax=None):
    mc_model = MCSampler(model, sample_size)

    return