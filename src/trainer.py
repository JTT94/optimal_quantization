
class Trainer(object):

    def __init__(self,
                 alpha_dist,
                 quantizer,
                 optimizer_map,
                 optimizer_atoms,
                 map_scheduler_func = None,
                 atoms_scheduler_func=None,
                 verbose=False):

        # get distribution to sample from
        self.alpha_dist = alpha_dist

        # optimizers
        self.optimizer_map = optimizer_map
        self.optimizer_atoms = optimizer_atoms

        # schedulers
        if map_scheduler_func is not None:
            self.scheduler_map = map_scheduler_func(self.optimizer_map)
        if atoms_scheduler_func is not None:
            self.scheduler_atoms = atoms_scheduler_func(self.optimizer_map)

        # quantizer
        self.quantizer = quantizer

        # verbose flag
        self.verbose = verbose

    def map_train_step(self, batch_size):
        # sample x
        xs = self.alpha_dist.sample(sample_shape=(batch_size,))

        # Clear gradients w.r.t. parameters
        self.optimizer_map.zero_grad()

        # Get dual objective to maximise
        dual_objective = self.quantizer.stochastic_dual_approx(xs)
        map_loss = -dual_objective

        # Getting gradients w.r.t. parameters
        map_loss.backward()

        # Updating parameters
        self.optimizer_map.step()

    def atoms_train_step(self, batch_size):
        # sample x
        xs = self.alpha_dist.sample(sample_shape=(batch_size,))

        # Clear gradients w.r.t. parameters
        self.optimizer_atoms.zero_grad()

        # Get loss objective to minimise
        atoms_loss = self.quantizer.stochastic_dual_approx(xs)

        # Getting gradients w.r.t. parameters
        atoms_loss.backward()

        # Updating parameters
        self.optimizer_atoms.step()

    def training_step(self, batch_size, n_sub_iters_map, n_sub_iters_atoms):
        for _ in range(n_sub_iters_map):
            self.map_train_step(batch_size)

        for _ in range(n_sub_iters_atoms):
            self.atoms_train_step(batch_size)

    def train(self, n_iters, batch_size, n_sub_iters_map, n_sub_iters_atoms, lr_steps):
        for iteration in range(n_iters):
            self.training_step(batch_size, n_sub_iters_map, n_sub_iters_atoms)

            if iteration % lr_steps == 0:
                # Decay Learning Rate
                self.scheduler_atoms.step()
                self.scheduler_map.step()
