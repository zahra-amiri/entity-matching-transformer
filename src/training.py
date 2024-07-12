import logging
import os
from data_loader import load_data, DataType

import torch
from tqdm import tqdm, trange
from data_representation import DeepMatcherProcessor
from logging_customized import setup_logging
from model import save_model
from tensorboardX import SummaryWriter
from prediction import predict

setup_logging()


def train(data_path,tokenizer, device,
          train_dataloader,
          model,
          optimizer,
          scheduler,
          evaluation,
          num_epocs,
          max_grad_norm,
          save_model_after_epoch,
          experiment_name,
          output_dir,
          model_type):
    logging.info("***** Run training *****")
    tb_writer = SummaryWriter(os.path.join(output_dir, experiment_name))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    processor = DeepMatcherProcessor()
    test_examples = processor.get_test_examples(data_path)

    test_data_loader = load_data(test_examples,
                                 processor.get_labels(),
                                 tokenizer,
                                 150,
                                 16,
                                 DataType.TEST,
                                 model_type=model_type)

    # we are interested in 0 shot learning, therefore we already evaluate before training.
    eval_results = evaluation.evaluate(model, device, -1)
    for key, value in eval_results.items():
        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

    with open(os.path.join(output_dir, experiment_name) + 'test.txt', 'w') as file:
        for epoch in trange(int(num_epocs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()

                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

                if model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                tr_loss += loss.item()

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                global_step += 1

                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
                logging_loss = tr_loss



            file.write("Test")
            simple_accuracy, f1, classification_report, predictions = predict(model, device, test_data_loader)
            file.write("Prediction done for {} examples.F1: {}, Simple Accuracy: {}".format(len(test_data_loader), f1, simple_accuracy))

            file.write(classification_report)

            file.write(predictions)


            eval_results = evaluation.evaluate(model, device, epoch)
            for key, value in eval_results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

            if save_model_after_epoch:
                save_model(model, experiment_name, output_dir, epoch=epoch)

    tb_writer.close()
