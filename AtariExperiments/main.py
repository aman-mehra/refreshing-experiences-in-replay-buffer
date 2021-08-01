from runner import *
from settings import *

trainer = nStepRunner(config)
trainer.setup()
trainer.learn_n_step()

