import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy



class MetaLearner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        # self.net = Learner(config, args.imgc, args.imgsz)
        
        self.env = gym.make(self.args.env)

        demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
        demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)

        logger.info('loading demos')
        self.train_demos = utils.load_demos(demos_path)
        logger.info('loaded demos')
        if args.episodes:
            if args.episodes > len(self.train_demos):
                raise ValueError("there are only {} train demos".format(len(self.train_demos)))
            self.train_demos = self.train_demos[:args.episodes]

        self.val_demos = utils.load_demos(demos_path_valid)
        if args.val_episodes > len(self.val_demos):
            logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.val_demos)))
        self.val_demos = self.val_demos[:self.args.val_episodes]

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        self.obss_preprocessor = utils.ObssPreprocessor(args.model, observation_space,
                                                        getattr(self.args, 'pretrained_model', None))

        # Define actor-critic model
        self.net = utils.load_model(args.model, raise_not_found=False)
        if self.net is None:
            if getattr(self.args, 'pretrained_model', None):
                self.net = utils.load_model(args.pretrained_model, raise_not_found=True)
            else:
                self.net = ACModel(self.obss_preprocessor.obs_space, action_space,
                                       args.image_dim, args.memory_dim, args.instr_dim,
                                       not self.args.no_instr, self.args.instr_arch,
                                       not self.args.no_mem, self.args.arch)
        self.obss_preprocessor.vocab.save()
        # utils.save_model(self.net, args.model)
        self.fast_net = copy.deepcopy(self.net)
        self.net.train()
        self.fast_net.train()

        if torch.cuda.is_available():
            self.net.cuda()
            self.fast_net.cuda()
        
        self.optimizer = torch.optim.Adam(lr= self.args.lr, eps=self.args.optim_eps)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)





    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward_batch(batch, net = 'fast'):
        if net == 'fast':
            acmodel = self.fast_net
        else:
            acmodel = self.net

        batch = utils.demos.transform_demos(batch)
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]

        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        flat_batch = np.array(flat_batch)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), acmodel.memory_size], device=self.device)

        preprocessed_first_obs = self.obss_preprocessor(obss[inds], device=self.device)
        instr_embedding = acmodel._get_instr_embedding(preprocessed_first_obs.instr)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished
                new_memory = acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], instr_embedding[:len(inds)])['memory']

            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory
            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]
        # Here, actual backprop upto args.recurrence happens
        final_loss = 0
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0

        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence
        for _ in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            action_step = action_true[indexes]
            mask_step = mask[indexes]
            model_results = acmodel(
                preprocessed_obs, memory * mask_step,
                instr_embedding[episode_ids[indexes]])
            dist = model_results['dist']
            memory = model_results['memory']

            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            indexes += 1

        final_loss /= self.args.recurrence

        # if is_training:
        #     self.optimizer.zero_grad()
        #     final_loss.backward()
        #     self.optimizer.step()

        log = {}
        log["entropy"] = float(final_entropy / self.args.recurrence)
        log["policy_loss"] = float(final_policy_loss / self.args.recurrence)
        log["accuracy"] = float(accuracy)

        return final_loss



    # def forward(self, x_spt, y_spt, x_qry, y_qry):
    def forward(self, demo):
        
        task_num = len(demo)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i], i is tasks idx
        accuracy = [0 for _ in range(self.update_step + 1)]
        
        # final_loss = forward_batch(demo)
        # grad = torch.autograd.grad(final_loss, self.net.parameters())
        self.fast_net = copy.deepcopy(self.net)
        
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        for i in range(task_num):
            self.fast_net.zero_grad()
            # 1. run the i-th task and compute loss for k=0
            self.optimizer.zero_grad()
            loss = self.forward_batch(demo[i],'fast')
            loss.backward()

            #grad = torch.autograd.grad(loss, self.net.parameters())
            self.optimizer(self.fast_net)
            #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                self.optimizer.zero_grad()
                loss = self.forward_batch(demo[i],'fast')
                loss.backward()
                # 2. compute grad on theta_pi
                # grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                self.optimizer(self.fast_net)
                loss_q = self.forward_batch(demo[i],fast_weights)
               # loss_q will be overwritten and just keep the loss_q on last update step.
                losses_q[k + 1] += loss_q

                            # 4. record last step's loss for task i
            losses_q.append(loss_q)

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # (torch.norm(p).item())
        self.meta_optim.step()



        return loss_q


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs

if __name__ == '__main__':
    main()
