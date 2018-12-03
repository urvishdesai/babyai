import torch
import torch.nn as nn
from imitation import ImitationLearning
import utils
import meta_utils

class MetaLearner(nn.Module):
    """ Bare Meta-learner class
        Should be added: intialization, hidden states, more control over everything
    """
    def __init__(self, model, args):
        super(MetaLearner, self).__init__()
        self.weights = Parameter(torch.Tensor(1, 2))
        self.args = args
        self._init_train(args)

    def forward(self, forward_model, backward_model):
        """ Forward optimizer with a simple linear neural net
        Inputs:
            forward_model: PyTorch module with parameters gradient populated
            backward_model: PyTorch module identical to forward_model (but without gradients)
              updated at the Parameter level to keep track of the computation graph for meta-backward pass
        """
        f_model_iter = get_params(forward_model)
        b_model_iter = get_params(backward_model)
        for f_param_tuple, b_param_tuple in zip(f_model_iter, b_model_iter): # loop over parameters
            # Prepare the inputs, we detach the inputs to avoid computing 2nd derivatives (re-pack in new Variable)
            (module_f, name_f, param_f) = f_param_tuple
            (module_b, name_b, param_b) = b_param_tuple
            inputs = Variable(torch.stack([param_f.grad.data, param_f.data], dim=-1))
            # Optimization step: compute new model parameters, here we apply a simple linear function
            dW = F.linear(inputs, self.weights).squeeze()
            param_b = param_b + dW
            # Update backward_model (meta-gradients can flow) and forward_model (no need for meta-gradients).
            module_b._parameters[name_b] = param_b
            param_f.data = param_b.data

    def _init_train(args):
        if args.multi_env is not None:
            assert len(args.multi_demos) == len(args.multi_episodes)

        args.model = args.model or ImitationLearning.default_model_name(args)
        utils.configure_logging(args.model)
        logger = logging.getLogger(__name__)

        self.il_learn_forward = ImitationLearning(args)
        self.il_learn_backward = ImitationLearning(args) # Change to initialize with shared params

        # Define logger and Tensorboard writer
        self.header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
                  + ["validation_accuracy"])
        if args.multi_env is None:
            self.header.extend(["validation_return", "validation_success_rate"])
        else:
            self.header.extend(["validation_return_{}".format(env) for env in args.multi_env])
            self.header.extend(["validation_success_rate_{}".format(env) for env in args.multi_env])
        writer = None
        if args.tb:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(utils.get_log_dir(args.model))

        # Define csv writer
        selof.csv_writer = None
        self.csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
        first_created = not os.path.exists(self.csv_path)
        # we don't buffer data going in the csv log, cause we assume
        # that one update will take much longer that one write to the log
        self.csv_writer = self.csv.writer(open(self.csv_path, 'a', 1))
        if first_created:
            self.csv_writer.writerow(self.header)

        # Get the status path
        self.status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')

        # Log command, availability of CUDA, and model
        logger.info(args)
        logger.info("CUDA available: {}".format(torch.cuda.is_available()))
        logger.info(self.il_learn.acmodel)

    def train():
        for meta_epoch in range(args.meta_epochs): # Meta-training loop (train the optimizer)
            optimizer.zero_grad()
            losses = []
            for inputs, labels in train_data:   # Meta-forward pass (train the model)
                il_learn.train(il_learn.train_demos, writer, csv_writer, status_path, header)


                forward_model.zero_grad()         # Forward pass
                inputs = Variable(inputs)
                labels = Variable(labels)
                output = forward_model(inputs)
                loss = loss_func(output, labels)  # Compute loss
                losses.append(loss)
                loss.backward()                   # Backward pass to add gradients to the forward_model
                optimizer(forward_model,          # Optimizer step (update the models)
                        backward_model)
            meta_loss = sum(losses)             # Compute a simple meta-loss
            meta_loss.backward()                # Meta-backward pass
            meta_optimizer.step()               # Meta-optimizer step



