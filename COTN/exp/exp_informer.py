from data.data_loader import data_provider
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

class Exp_Informer(Exp_Basic):
    def __init__(self, config):
        self.config = config  # 先设置config属性
        super(Exp_Informer, self).__init__(config)  # 再调用父类初始化
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        
        if self.config.model=='informer' or self.config.model=='informerstack':
            e_layers = self.config.e_layers if self.config.model=='informer' else self.config.s_layers
            model = model_dict[self.config.model](
                self.config.enc_in,
                self.config.dec_in, 
                self.config.c_out, 
                self.config.seq_len, 
                self.config.label_len,
                self.config.pred_len, 
                self.config.factor,
                self.config.d_model, 
                self.config.n_heads, 
                e_layers,
                self.config.d_layers, 
                self.config.d_ff,
                self.config.dropout, 
                self.config.attn,
                self.config.embed,
                self.config.freq,
                self.config.activation,
                self.config.output_attention,
                self.config.distil,
                self.config.mix,
                self.device
            ).float()
        
        if self.config.use_multi_gpu and self.config.use_gpu:
            model = nn.DataParallel(model, device_ids=self.config.device_ids)
        return model

    def _get_data(self, flag):
        """使用新的数据提供器"""
        data_set, data_loader = data_provider(self.config, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.config.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(10):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{10}')
            
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_bar):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:,-self.config.pred_len:,:]).float()
                dec_inp = torch.cat([batch_y[:,:self.config.label_len,:], dec_inp], dim=1).float().to(self.device)

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.config.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.config.features=='MS' else 0
                        batch_y = batch_y[:,-self.config.pred_len:,f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.config.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.config.features=='MS' else 0
                    batch_y = batch_y[:,-self.config.pred_len:,f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.config.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.config.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                train_bar.set_postfix({'loss': '{:.6f}'.format(loss.item())})

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.config)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        """
        Enhanced test function with comprehensive metrics and visualization
        Loads the best checkpoint and generates detailed results
        """
        # Load the best model checkpoint
        path = os.path.join(self.config.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        
        if not os.path.exists(best_model_path):
            print(f"❌ Checkpoint not found: {best_model_path}")
            print("Please train the model first or check the setting name.")
            return None
            
        print(f"📁 Loading checkpoint from: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        preds = []
        trues = []
        
        print(f"🧪 Running test inference...")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(test_loader)} batches")

        preds = np.array(preds)
        trues = np.array(trues)
        print('Raw test shape:', preds.shape, trues.shape)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('Reshaped test shape:', preds.shape, trues.shape)

        # Calculate basic metrics (for backward compatibility)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"📊 Basic Metrics:")
        print(f"   MAE: {mae:.6f}")
        print(f"   MSE: {mse:.6f}")  
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAPE: {mape:.6f}")
        print(f"   MSPE: {mspe:.6f}")

        # Calculate comprehensive metrics
        from utils.metrics import comprehensive_metrics
        comp_metrics = comprehensive_metrics(preds, trues)
        print(f"\n📊 Comprehensive Metrics:")
        for metric_name, value in comp_metrics.items():
            # Convert numpy arrays to scalar values with better handling
            try:
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    else:
                        value = float(np.mean(value))
                elif hasattr(value, 'item'):
                    value = float(value.item())
                else:
                    value = float(value)
            except (ValueError, TypeError):
                value = float(np.mean(np.array(value).flatten()))
                
            if metric_name in ['MAPE', 'SMAPE', 'WAPE']:
                print(f"   {metric_name}: {value:.4f}%")
            else:
                print(f"   {metric_name}: {value:.6f}")

        # Create results directory
        base_folder = './test_results/' + setting
        folder_path = base_folder + '/'
        counter = 1
        
        # Create unique folder if exists
        while os.path.exists(folder_path):
            folder_path = f"{base_folder}_run{counter}/"
            counter += 1
            
        os.makedirs(folder_path, exist_ok=True)
        print(f"📂 Results will be saved to: {folder_path}")

        # Save original format files (for backward compatibility)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Generate comprehensive test visualizations and results
        try:
            from utils.test_visualization import (
                plot_prediction_comparison, plot_comprehensive_metrics, 
                save_test_results, generate_test_report
            )
            
            dataset_name = self.config.data
            print(f"\n📊 Generating comprehensive test results...")
            
            # Create visualization plots
            print("  📈 Creating prediction comparison plots...")
            plot_prediction_comparison(preds, trues, dataset_name, setting, 
                                     output_dir="./test_plots", show_samples=5, 
                                     pred_len=self.config.pred_len)
            
            print("  📊 Creating comprehensive metrics analysis...")
            plot_comprehensive_metrics(preds, trues, dataset_name, setting, 
                                     output_dir="./test_plots")
            
            # Save detailed results
            print("  💾 Saving detailed test results...")
            save_test_results(preds, trues, dataset_name, setting, 
                            output_dir="./test_exports")
            
            # Generate comprehensive report
            print("  📄 Generating test report...")
            generate_test_report(dataset_name, setting, 
                                folder_path, "./test_plots", "./test_exports")
            
            print(f"\n✅ Comprehensive test analysis completed!")
            print(f"   📁 Test results: {folder_path}")
            print(f"   📈 Visualizations: ./test_plots/")
            print(f"   📊 Export files: ./test_exports/")
            
        except Exception as e:
            print(f"⚠️  Error generating comprehensive results: {e}")
            import traceback
            traceback.print_exc()

        return {
            'preds': preds,
            'trues': trues,
            'basic_metrics': {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe},
            'comprehensive_metrics': comp_metrics,
            'results_path': folder_path
        }

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.config.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.config.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.config.pred_len, batch_y.shape[-1]]).float()
        elif self.config.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.config.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.config.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                if self.config.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.config.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.config.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.config.features=='MS' else 0
        batch_y = batch_y[:,-self.config.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
