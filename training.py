import torch
import torch.nn as nn
import torch.nn.functional as F
from egg.core import NTXentLoss, TopographicSimilarity

from config import Config
from data import DataLoader, collate_only_features, collate_with_distractors
from learning_experiment import LearningExp
from measures import binary_accuracy, generalization_score, mean_production_similarity
from preprocessing import Preprocessor

try:
    import wandb

    WANDB_IS_AVAILABLE = True
except ImportError:
    print("WandB logging not available, pip install wandb?")
    WANDB_IS_AVAILABLE = False


class Trainer(object):

    """Trainer for a Language Learning experiment"""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        learningexp: LearningExp,
        batch_size: int = 5,
        eval_batch_size: int = 30,
        lr: float = 1e-3,
        con_weight: float = 0.1,
        logging_freq: int = 1,
        use_wandb: bool = True,
        outfile: str = None,
        seed=None,
    ):
        """Initialize the trainer object

        :config: a configuration object
        :model: the model to train
        :learningexp: the learning experiment to replicate
        :batch_size: the batch size, where applicable
        :logging_freq: logging frequency in epochs (default 1
        :lr: learning rate
        :use_wandb: whether to use weights and biases logging
        :outfile: path to output file

        """
        self.learningexp = learningexp
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.preprocess = Preprocessor(config, learningexp.lang)
        print(learningexp)

        self.vocab_size = len(config.alphabet) + 1

        self._epoch = 0
        self._step = 0
        self.logging_freq = logging_freq

        self.con_weight = con_weight

        self.use_wandb = WANDB_IS_AVAILABLE and use_wandb

        # Prepare results format and output directory
        print(f"Will store outputs to {outfile}")
        self.outfile = outfile
        self.output_log = LearningExp.empty_like(learningexp)
        self.output_log.info["Participant ID"] = seed

        self.seed = seed
        self._prepare_test_data()

    def _prepare_test_data(self):
        """Prepare the test data"""
        # Memorization Test data
        self._raw_mem_test_data = self.learningexp.get_memorization_test_data()
        self._prep_mem_test_data = self.preprocess(self._raw_mem_test_data)

        # Regularization Test data
        self._raw_reg_test_data = self.learningexp.get_regularization_test_data()
        self._prep_reg_test_data = self.preprocess(self._raw_reg_test_data)

    def _generative_step(self, batch):
        """Produce a batch of messages for a batch of inputs"""
        outputs = self.model(sender_input=batch.features)
        # RnnSenderGs outputs [bsz, seqlen, vocab_size]

        # print("OUTPUTS", outputs)
        # print("OUTPUTS.shape", outputs.shape)
        target_onehot = F.one_hot(
            batch.target_word, num_classes=self.vocab_size
        ).float()
        gen_loss = F.binary_cross_entropy(outputs, target_onehot)
        return gen_loss

    def _contrastive_step(self, batch):
        # Process scene
        encoded_input = self.model.input2hidden(batch.features)
        # Process msg
        encoded_message = self.model(message=batch.target_word)

        # push f(scene) and g(msg) together
        con_loss, con_acc = NTXentLoss.ntxent_loss(
            encoded_input, encoded_message, temperature=1.0, similarity="cosine"
        )
        con_loss = con_loss.mean()
        con_acc = con_acc["acc"].mean()
        return con_loss, con_acc

    def _dicriminative_step(self, batch):
        assert len(batch) == 1, "Discriminative step can only do batch size 1"

        batch_size = 1

        temperature = 1.0

        h_input = self.model.input2hidden(batch.features)
        h_distractors = self.model.input2hidden(batch.distractors.squeeze())

        h_inp_and_dis = torch.cat([h_input, h_distractors], dim=0)

        h_target = self.model(message=batch.target_word)

        similarity_f = torch.nn.CosineSimilarity(dim=1)

        logits = similarity_f(h_inp_and_dis, h_target) / temperature
        # correct similarity should be at pos 0
        # sim = [target~inp, target~dis1, target~dis2, target~dis3,...]
        logits = logits.unsqueeze(0)

        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)

        dis_loss = F.cross_entropy(logits, labels)
        acc = (torch.argmax(logits.detach(), dim=1) == labels).float().detach()

        return dis_loss, acc

    def _log_loss(
        self,
        task,
        epoch,
        loss,
        gen_loss=0.0,
        con_loss=0.0,
        con_acc=0.0,
        dis_loss=0.0,
        dis_acc=0.0,
    ):
        task_abbr = task[:4] + "."
        print(
            f"[{task_abbr} / Epoch {self._epoch:4d} / Block-Epoch {epoch:2d} / Step {self._step:7d}] train/loss: {loss:.4f} | train/gen_loss: {gen_loss:.4f} | train/con_loss: {con_loss:.4f} | train/con_acc: {con_acc:.4f} | train/dis_loss={dis_loss:.4f} | train/dis_acc={dis_acc:.4f}"
        )

    def train_exposure(self, data, num_epochs=1):
        """
        Passive exposure to predict `target` given `shape` and `angle`
        (discriminative)
        """

        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_gen_loss = 0.0
            total_con_loss = 0.0
            total_con_acc = 0.0

            self.model.train()
            for i, batch in enumerate(loader):
                if torch.cuda.is_available():
                    batch = batch.cuda()
                self.optimizer.zero_grad()
                # print("INPUTS", batch.features)

                ### GENERATIVE STEP ###
                gen_loss = self._generative_step(batch)

                ### CONTRASTIVE STEP ###
                if self.con_weight > 0.0:
                    # Only compute contrastive loss, if nonzero weight
                    con_loss, con_acc = self._contrastive_step(batch)

                    loss = gen_loss + self.con_weight * con_loss
                else:
                    # Only generative loss in this case
                    loss = gen_loss

                    # Compatibility with logging
                    con_loss = torch.Tensor([0])
                    con_acc = torch.Tensor([0])  

                loss.backward()
                self.optimizer.step()
                self._step += 1
                if self.use_wandb:
                    wandb.log(
                        {
                            "train/con_loss": con_loss.item(),
                            "train/gen_loss": gen_loss.item(),
                            "train/con_acc": con_acc.item(),
                            "train/loss": loss.item(),
                        }
                    )

                total_loss += loss.item() * len(batch)
                total_gen_loss += gen_loss.item() * len(batch)
                total_con_loss += con_loss.item() * len(batch)
                total_con_acc += con_acc.item() * len(batch)

            if epoch % self.logging_freq == 0:
                avg_total_loss = total_loss / len(data)
                avg_gen_loss = total_gen_loss / len(data)
                avg_con_loss = total_con_loss / len(data)
                avg_con_acc = total_con_acc / len(data)
                self._log_loss(
                    "Exposure",
                    epoch,
                    avg_total_loss,
                    gen_loss=avg_gen_loss,
                    con_loss=avg_con_loss,
                    con_acc=avg_con_acc,
                )

            # if epoch % eval_freq == 0:
            #     acc = self.eval_production(data)
            #     print(f"[Epoch {epoch}] train/acc: {acc*100:.2f}%")

    def train_guessing(self, data, num_epochs=1):
        """Guess right object among distractors
        (constrastive OR generative)"""
        self.model.train()
        # Keep Batch Size 1 to stack distractors
        loader = DataLoader(
            data, batch_size=1, shuffle=True, collate_fn=collate_with_distractors
        )

        N = len(data)

        for epoch in range(1, num_epochs + 1):
            total_gen_loss = 0.0
            total_dis_loss = 0.0
            total_loss = 0.0
            total_acc = 0.0
            for batch in loader:
                # batch.features [1,6]
                # batch.distractors [1,3,6]
                if torch.cuda.is_available():
                    batch = batch.cuda()
                self.optimizer.zero_grad()
                dis_loss, dis_acc = self._dicriminative_step(batch)
                dis_acc = dis_acc.mean()
                gen_loss = self._generative_step(batch)
                loss = gen_loss + self.con_weight * dis_loss

                loss.backward()

                self.optimizer.step()
                self._step += 1

                if self.use_wandb:
                    wandb.log(
                        {
                            "train/gen_loss": gen_loss.item(),
                            "train/dis_loss": dis_loss.item(),
                            "train/dis_acc": dis_acc.item(),
                            "train/loss": loss.item(),
                        }
                    )

                total_gen_loss += gen_loss.item()
                total_dis_loss += dis_loss.item()
                total_loss += loss.item()
                total_acc += dis_acc.item()

            if epoch % self.logging_freq == 0:
                self._log_loss(
                    "Guessing",
                    epoch,
                    total_loss / N,
                    gen_loss=total_gen_loss / N,
                    con_loss=total_dis_loss / N,
                    con_acc=total_acc / N,
                )

    def train_production(self, data, num_epochs=1):
        """Produce target given shape and angle"""
        self.model.train()
        N = len(data)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for batch in loader:
                if torch.cuda.is_available():
                    batch = batch.cuda()
                self.optimizer.zero_grad()
                ### GENERATIVE STEP ###
                gen_loss = self._generative_step(batch)
                gen_loss.backward()
                self.optimizer.step()
                self._step += 1

                if self.use_wandb:
                    wandb.log(
                        {
                            "train/gen_loss": gen_loss.item(),
                            "train/loss": gen_loss.item(),
                        }
                    )

                total_loss += gen_loss.item() * len(batch)

            if epoch % self.logging_freq == 0:
                loss = total_loss / N
                self._log_loss("Production", epoch, loss, gen_loss=loss)

    def generate_messages(self, data):
        """Generate messages for inputs"""
        self.model.eval()
        loader = DataLoader(
            data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collate_only_features,
        )

        all_messages = []
        with torch.no_grad():
            for batch in loader:
                if torch.cuda.is_available():
                    batch = batch.cuda()
                outputs = self.model(sender_input=batch.features)
                batch_messages = torch.argmax(outputs, dim=-1)
                all_messages.append(batch_messages)
        return torch.cat(all_messages, dim=0)

    def get_target_words(self, data):
        """Reconstruct target words from encoded words,
        TODO: cant we just use the non-preprocessed words instead??
        """
        return self.preprocess.tokenizer.decode_batch([obj.target_word for obj in data])

    def test_memorization(self, data):
        """Test memorization"""
        messages = self.generate_messages(data)
        messages_str = self.preprocess.tokenizer.decode_batch(messages)
        target_words = self.get_target_words(data)
        print(
            "MEMORIZATION (generated, target):", list(zip(messages_str, target_words))
        )
        acc, correct = binary_accuracy(messages_str, target_words)

        prodsim = mean_production_similarity(messages_str, target_words)

        mem_scores = {"mem/acc": acc, "mem/prodsim": prodsim}

        ## TODO add outputs to log
        # Save memorization phase data
        self.output_log.append_results(
            self._epoch,
            "MemorizationTest",
            self._raw_mem_test_data,
            messages_str,
            correct_messages=correct,
            producer=self.seed,
        )

        return messages_str, mem_scores

    def test_regularization(
        self,
        data,
        familiar_scenes=None,
        messages_for_familiar_scenes=None,
        new_scenes=None,
        messages_for_new_scenes=None,
    ):
        """Test memorization
        TODO: remove arguments and fill them here
        """
        self.model.eval()
        messages = self.generate_messages(data)
        meanings = torch.stack([obj.features for obj in data])
        topsim = TopographicSimilarity.compute_topsim(
            meanings.cpu(),
            messages.cpu(),
            meaning_distance_fn="cosine",
            message_distance_fn="edit",
        )
        reg_scores = {"reg/topsim": topsim}

        messages_str = self.preprocess.tokenizer.decode_batch(messages)

        if messages_for_new_scenes is not None:
            print(
                "REGULARIZATION (generated, human):",
                list(zip(messages_str, messages_for_new_scenes)),
            )

        self.output_log.append_results(
            self._epoch,
            "RegularizationTest",
            self._raw_reg_test_data,
            messages_str,
            correct_messages=None,  # who knows
            producer=self.seed,
        )

        if familiar_scenes is not None:
            assert len(messages_for_familiar_scenes) == len(familiar_scenes)
            assert new_scenes is not None
            assert len(new_scenes) == len(data)

            # decode from int's to string to account for EOS

            gen_score, __gen_score_pval = generalization_score(
                familiar_scenes, messages_for_familiar_scenes, new_scenes, messages_str
            )

            reg_scores["reg/genscore"] = gen_score

        if messages_for_new_scenes is not None:
            reg_scores["reg/prodsim"] = mean_production_similarity(
                messages_str, messages_for_new_scenes
            )

        return messages_str, reg_scores

    def evaluate(self):
        """Runs memorization test and regularization test"""
        # MEMORIZATION
        mem_messages, mem_scores = self.test_memorization(self._prep_mem_test_data)

        # GENERALIZATION
        familiar_scenes = self.learningexp.get_memorization_scenes()
        new_scenes = self.learningexp.get_regularization_scenes()
        human_messages_for_new_scenes = self._raw_reg_test_data["Input"]

        __reg_messages, reg_scores = self.test_regularization(
            self._prep_reg_test_data,
            familiar_scenes=familiar_scenes,
            messages_for_familiar_scenes=mem_messages,
            new_scenes=new_scenes,
            messages_for_new_scenes=human_messages_for_new_scenes,
        )

        scores = {**mem_scores, **reg_scores}
        scores["Epoch"] = self._epoch
        if self.use_wandb:
            wandb.log(scores)
        print(f"[Epoch {self._epoch:4d} / step {self._step:7d}] {scores}")
        return scores

    def train_exactly_as_humans(self, num_iterations=100, epochs_per_block=1):
        """
        Train 3 times per iteration on exposure, guessing, and production
        with same data subsets as humans
        """
        ### Preprocessing ###
        ### Training data
        exposure_1_data = self.preprocess(self.learningexp.get_exposure_data(1))
        exposure_2_data = self.preprocess(self.learningexp.get_exposure_data(2))
        exposure_3_data = self.preprocess(self.learningexp.get_exposure_data(3))

        guessing_1_data = self.preprocess(self.learningexp.get_guessing_data(1))
        guessing_2_data = self.preprocess(self.learningexp.get_guessing_data(2))
        guessing_3_data = self.preprocess(self.learningexp.get_guessing_data(3))

        production_1_data = self.preprocess(self.learningexp.get_guessing_data(1))
        production_2_data = self.preprocess(self.learningexp.get_guessing_data(2))
        production_3_data = self.preprocess(self.learningexp.get_guessing_data(3))

        for __ in range(1, num_iterations + 1):
            # Round 1
            self.train_exposure(exposure_1_data, num_epochs=epochs_per_block)
            self.train_guessing(guessing_1_data, num_epochs=epochs_per_block)
            self.train_production(production_1_data, num_epochs=epochs_per_block)

            # Round 2
            self.train_exposure(exposure_2_data, num_epochs=epochs_per_block)
            self.train_guessing(guessing_2_data, num_epochs=epochs_per_block)
            self.train_production(production_2_data, num_epochs=epochs_per_block)

            # Round 3
            self.train_exposure(exposure_3_data, num_epochs=epochs_per_block)
            self.train_guessing(guessing_3_data, num_epochs=epochs_per_block)
            self.train_production(production_3_data, num_epochs=epochs_per_block)

            self._epoch += 1
            # Test
            self.evaluate()

        if self.outfile is not None:
            self.output_log.save(self.outfile)

            # if self.use_wandb:
            #     artifact = wandb.Artifact(self.outfile,
            #             "output",
            #             description="Model outputs in participant-file format",
            #             metadata=self.output_log.info)
            #     artifact.add_file(self.outfile)
            #     wandb.log_artifact(artifact)

    def train(self, num_iterations=1000, epochs_per_block=1):
        """Train in the same way (exposure-style) on all data"""
        # train_data = self.preprocess(self.learningexp.get_all_training_data())
        # Changed to only exposure 3 data, as it contains everything uniq 2022-05-21, lgalke
        train_data = self.preprocess(self.learningexp.get_exposure_data(3))

        for __ in range(1, num_iterations + 1):
            # Train (exposure strategy on all data)
            self.train_exposure(train_data, num_epochs=epochs_per_block)

            # Don't do that, here
            # self.train_guessing(guessing_loader)
            # self.train_production(production_loader)

            # Test
            self._epoch += 1
            self.evaluate()

        if self.outfile is not None:
            self.output_log.save(self.outfile)
