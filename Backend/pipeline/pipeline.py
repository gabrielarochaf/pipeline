
import os
import pickle
import pprint
import warnings
import json
from sklearn import discriminant_analysis

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import cv2


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from collections import Counter, OrderedDict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from IPython.display import clear_output
from sklearn.metrics import confusion_matrix

from routine import Routine
from utils import Utils
from sample import Sample



class Pipeline:

    def __init__(self, data_folder: str, samples: dict):
        self.folder = data_folder
        self.samples = samples
        self.routine = Routine()
        self.utils = Utils()

        self.properties = {'fontproperties': {'family': 'sans-serif',
                                              'weight': 'bold',
                                              'size': 10},
                           }

    def _remove_bg_sample(self, sample: Sample,
                          dark_prefix='DARKREF_',
                          white_prefix='WHITEREF_', ):
        pass

    def _signal_filter(self, sample: Sample,
                       order=2, window=21, dv=1, mode='constant'):



        matrix = self.routine.hsi2matrix(sample.normalized)
        matrix = self.routine.normalize_mean(matrix)
        #matrix = self.routine.sgolay(matrix=matrix, order=order, window=window, derivative=dv, mode=mode)

        #return self.routine.snv(matrix=matrix)
        return matrix


    def visualize_images(self, folder, samples, normalized_data, sample_cluster, SAVE_NAME):
        """
        A função "visualize_images" visa plotar as bactérias.

            - Realiza carregamento das bactérias, utlizando a função "load_bacterias" explicado em Utils.
            - Setando o tamanho em linhas a ser plotado de cada imagem
            - Converte-a para RGB para que possa ser visualizada


            Retorno:
            - Plot da imagem da bacteria em RGB
        """

        for idx, sample in enumerate(list(samples.keys())):
            # data = Utils.load_hsi_sample(path=folder, name=sample)

            image = normalized_data[80, :, :]
            out_i = self.routine.getCluster(image, sample_cluster, 0, (1, 1, 1))
            # fig = plt.figure()
            # plt.imshow(rgbscale(out_i))
            # plt.savefig(f"{SAVE_NAME}_1.png")  # AQUI SALVA A IMAGEM
            cv2.imwrite(f"{SAVE_NAME}.png", self.routine.rgbscale(out_i))
            # plt.close(fig)
            # plt.show()


    def __concatenate_groups(self, spectral_range=(1, 241), case=0,
                             true_labels=None,
                             mean=False,
                             process=True):



        targets = {}
        targets_0 = {}
        concatenated = {}
        for idx, sample in enumerate(list(self.samples)):

            bacteria = Utils.load_hsi_sample(path=self.folder, name=sample)

            if process:
                matrix = self._signal_filter(sample=bacteria)
            else:
                matrix = self.routine.hsi2matrix(bacteria.normalized)
                matrix = self.routine.normalize_mean(matrix)

            # print('bacteria sample cluster', bacteria.sample_cluster)
            ind, _ = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))
            # print('ind', ind)
            matrix = matrix[ind, spectral_range[0]:spectral_range[1]]

            targets_0[self.samples[sample][0]] = self.samples[sample][case]

            if case == 0:
                targets[self.samples[sample][0]] = '_'.join(sample.split('_')[:2])
            else:
                if not true_labels:
                    print('true_labels must be specified')

                for keys in list(true_labels.keys()):
                    if true_labels[keys] == self.samples[sample][case]:
                        targets[self.samples[sample][case]] = keys

                        break

            if self.samples[sample][case] not in list(concatenated.keys()):
                concatenated[self.samples[sample][case]] = \
                    np.array([]).reshape(0, matrix.shape[1])

            if not mean:
                concatenated[self.samples[sample][case]] = \
                    np.concatenate((concatenated[self.samples[sample][case]], matrix), )
            else:
                concatenated[self.samples[sample][case]] = \
                    np.concatenate((concatenated[self.samples[sample][case]],
                                    self.routine.mean_from_2d(matrix, axis=0).reshape(1, -1)))

        return concatenated, (targets, targets_0)

    def get_group_mean_matrix(self, spectral_range=(1, 241), case=0, true_labels=None, process=True):


        concatenated, targets = self.__concatenate_groups(spectral_range=spectral_range,
                                                          case=case, true_labels=true_labels, process=process)

        mean_group = {}
        for key in list(concatenated.keys()):
            mean_group[key] = self.routine.mean_from_2d(matrix=concatenated[key], axis=0).reshape(1, -1)
        print(targets)
        return mean_group, targets

    def get_pca_matrix(self, spectral_range=(1, 241), case=0, true_labels=None, mean=False, pcs=3):


        targets = []
        mean_matrix = []
        if not mean:
            concatenated, targets_groups = self.__concatenate_groups(spectral_range=spectral_range,
                                                                     case=case, true_labels=true_labels)
        else:
            concatenated, targets_groups = self.__concatenate_groups(spectral_range=spectral_range,
                                                                     case=case, true_labels=true_labels, mean=True)

            for key, target in zip(list(concatenated.keys()), list(targets_groups[0].keys())):
                mean_matrix.append(concatenated[key])
                targets.append(target)

        var_exp = {}
        pca_matrix = {}
        pca = PCA(n_components=pcs)

        if not mean:
            all_samples = np.array([]).reshape((0, concatenated[list(concatenated.keys())[0]].shape[1]))
            for key in list(concatenated.keys()):
                all_samples = np.concatenate((all_samples, concatenated[key]))

            tmp_pcamatrix = pca.fit_transform(all_samples)
            var_exp['unique'] = pca.explained_variance_ratio_

            a = -1
            for key in list(concatenated.keys()):
                pca_matrix[key] = tmp_pcamatrix[a + 1:a + concatenated[key].shape[0], :]
                a = a + a + concatenated[key].shape[0]

        else:
            tmp_ = np.array(mean_matrix)
            tmp_ = tmp_.reshape(tmp_.shape[0] * tmp_.shape[1], -1)

            pcs = pca.fit_transform(tmp_)
            var_exp['unique'] = pca.explained_variance_ratio_

            for idx, key in enumerate(list(targets_groups[1].keys())):
                pca_matrix[key] = pcs[idx, :]

        return pca_matrix, var_exp, targets_groups

    def __get_samples(self, spectral_range=(1, 241), train=False):

        x = {}
        for idx, sample in enumerate(list(self.samples.keys())):
            bacteria = Utils.load_hsi_sample(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            ind, _ = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))

            idx_train, idx_test = train_test_split(ind, test_size=0.5, shuffle=False)
            x[sample] = matrix[idx_train if train else idx_test, spectral_range[0]:spectral_range[1]]

        return x

    def __rev_dict(self, samples_dict, case=0):
        rev = dict()
        for key in list(samples_dict.keys()):
            rev[samples_dict[key][case]] = key

        return rev

    def generate_results(self, models_name: str, work_dir: str, output_dir: str,
                         true_labels_name=None):

        x = self.__get_samples()
        models = load(models_name)

        results_models = pd.DataFrame(columns=[model.__class__.__name__ + '_' +
                                               str(idx) for idx, model in enumerate(models)],
                                      index=[key for key in x.keys()])

        if not true_labels_name:
            with open(os.path.join(work_dir, 'config.json'), 'r') as f_cfg:
                cfg = json.load(f_cfg)
                f_cfg.close()

        true_labels_name = self.__rev_dict(cfg['samples_training'])

        for idx, model in enumerate(models):
            print(model.__class__.__name__, idx)

            for idx_sample, key in enumerate(x.keys()):
                sample = x[key]
                y_hat = model.predict(sample)

                results_dict = dict()
                results = Counter(y_hat)

                for key_r in results.keys():
                    target = true_labels_name[int(key_r)]
                    results_dict[target] = results[key_r] / y_hat.shape[0]

                results_dict = OrderedDict(sorted(results_dict.items(), key=lambda t: x[1], reverse=True))
                rst_string = ' \n'.join(["{}: {:.4f}".format(key, results_dict[key])
                                         for key in list(results_dict.keys())[:1]])

                results_models.loc[key, model.__class__.__name__ + '_' + str(idx)] = rst_string

        results_models.to_csv(output_dir)

    def plot_pca_samples(self, pca_matrix_groups: dict, exp_var_groups: dict,
                         true_labels: tuple, out_dir: str, pc_plot=None, file_name='pca_matrix.jpeg'):


        plots = []
        if pc_plot is None:
            pc_plot = [[0, 1], [1, 2], [0, 2]]

        for xy in pc_plot:
            fig, ax = plt.subplots(**{'figsize': (12, 6), 'dpi': 100})
            plt.rcParams.update({'font.size': 10})

            for idx, group in enumerate(list(true_labels[1].keys())):
                plotargs = {'color': self.utils.colors[str(true_labels[1][group])],
                            'label': str(true_labels[1][group]) + '_{}'.format(
                                true_labels[0][true_labels[1][group]]),
                            'linewidth': 2}

                var_exp_keys = list(exp_var_groups.keys())
                ax.set_xlabel('PC{} ({:.2f}%)'.format(xy[0] + 1,
                                                      exp_var_groups[var_exp_keys[0]][xy[0]] * 100))
                ax.set_ylabel('PC{} ({:.2f}%)'.format(xy[1] + 1,
                                                      exp_var_groups[var_exp_keys[0]][xy[1]] * 100))

                ax.set_xlabel(ax.get_xlabel(), self.properties['fontproperties'])
                ax.set_ylabel(ax.get_ylabel(), self.properties['fontproperties'])

                if len(pca_matrix_groups[group].shape) == 1:
                    pca_matrix_groups[group] = pca_matrix_groups[group].reshape(1, -1)

                ax.text(pca_matrix_groups[group][:, xy[0]] + 0.04,
                        pca_matrix_groups[group][:, xy[1]], str(idx),
                        color="black", fontsize=6)

                plots.append(ax.plot(pca_matrix_groups[group][:, xy[0]],
                                     pca_matrix_groups[group][:, xy[1]],
                                     '.', **plotargs))

            labels = [plot[0].get_label() for plot in plots]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                legend = fig.legend(
                    plots,
                    labels,
                    loc='upper right',
                    bbox_to_anchor=(0.98, 2, 0.32, -0.102),
                    mode='expand',
                    ncol=2,
                    bbox_transform=fig.transFigure,
                )

            legend.set_visible(False)
            if out_dir:
                # figure_name = os.path.join(out_dir, 'PC_{}xPC_{}.jpeg'.format(xy[0] + 1, xy[1] + 1))
                # os.makedirs(out_dir, exist_ok=True)
                fig.savefig(file_name,
                            bbox_extra_artists=[],
                            bbox_inches='tight')

            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width + 3, legend_bbox.height))
            legend_squared = legend_ax.legend(
                *ax.get_legend_handles_labels(),
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=legend_fig.transFigure,
                frameon=False,
                fontsize=12,
                fancybox=None,
                shadow=False,
                ncol=1,
                mode='expand',
            )

            legend_ax.axis('off')

            if out_dir:
                # Save the legend as a separate figure
                # legend_figpath = 'legend_PC_{}xPC_{}.jpeg'.format(xy[0] + 1, xy[1] + 1)
                legend_fig.savefig(
                    # os.path.join(out_dir, '_LEGEND_' + file_name),
                    out_dir+'-6-RLEGEND_pca_graph',
                    bbox_inches='tight',
                    bbox_extra_artists=[legend_squared],
                )

    def plot_spectres(self, mean_group: dict, true_labels: dict,
                      out_dir: str, file_name='mean_spectres.jpeg', spectral_range=(1, 241)):



        plots = []
        fig, axes = plt.subplots(figsize=(12, 6), dpi=1000)
        plt.rcParams.update({'font.size': 10})

        wavelengths = self.routine.get_wavelength(folder=self.folder,
                                                  sample=list(self.samples.keys())[0],
                                                  spectral_range=spectral_range)

        axes.set_ylabel("Pseudo Absorbance")
        axes.set_xlabel("Wavelength (nm)")
        axes.set_xlabel(axes.get_xlabel(), self.properties['fontproperties'])
        axes.set_ylabel(axes.get_ylabel(), self.properties['fontproperties'])

        # for i, target in zip(range(mean_matrix.shape[0]), targets):
        for idx, group in enumerate(list(true_labels[0].keys())):
            for sample in range(mean_group[group].shape[0]):
                plotargs = {'color': self.utils.colors[str(group)],
                            'label': true_labels[0][group], 'linewidth': 2}

                ind = np.linspace(0, wavelengths.shape[1] - 1, num=5, dtype=int)

                axes.set_xticks(ind)
                axes.set_xticklabels(wavelengths[0, ind])

                plots.append(axes.plot(np.arange(mean_group[group].shape[1]),
                                       mean_group[group][sample, :], **plotargs))

        labels = [plot[0].get_label() for plot in plots]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legend = fig.legend(
                plots,
                labels,
                loc='upper right',
                bbox_to_anchor=(0.98, 2, 0.32, -0.102),
                mode='expand',
                ncol=2,
                bbox_transform=fig.transFigure,
            )

        legend.set_visible(False)
        # if out_dir:
        #     os.makedirs(out_dir, exist_ok=True)
        fig.savefig(file_name,
                    bbox_extra_artists=[],
                    bbox_inches='tight')

        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width + 3, legend_bbox.height))
        legend_squared = legend_ax.legend(
            *axes.get_legend_handles_labels(),
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=legend_fig.transFigure,
            frameon=False,
            fontsize=12,
            fancybox=None,
            shadow=False,
            ncol=1,
            mode='expand',
        )

        legend_ax.axis('off')

        # Save the legend as a separate figure
        legend_fig.savefig(
            out_dir+'-4-LEGEND_espectres',
            bbox_inches='tight',
            bbox_extra_artists=[legend_squared],
        )

    def process_images(self, SAVE_NAME: str):

        for idx, sample in enumerate(list(self.samples.keys())):
          save_name_temp=SAVE_NAME+'-1-'+sample
          print(sample, idx)
          bacteria = Sample(self.folder, sample)
          darkref = Sample(self.folder, sample, sample_prefix='DARKREF_')
          whiteref = Sample(self.folder, sample, sample_prefix='WHITEREF_')

          bacteria.normalized = self.routine.raw2mat(image=bacteria, dark=darkref, white=whiteref, inplace=False)
          #bacteria.normalized = bacteria.normalized[:,250:1500,:]

           # Pre process block
          matrix = self._signal_filter(sample=bacteria)

          rows, cols = bacteria.normalized.shape[1:]
          cube = self.routine.matrix2hsi(matrix, rows, cols)

          idx = self.routine.removeBg(cube, 2) + 1

          image = cube[80, :, :]

       
          out_i = self.routine.getCluster(image, idx, 1, (0, 0, 1))
          cluster = 1
          out_i = self.routine.rgbscale(out_i)
        # out_i2 = getCluster(image, idx, 2, (0, 1, 0))

          a = int(out_i.shape[0] / 2 * 0.8)
          b = int(out_i.shape[1] / 2 * 0.8)
          c = int(out_i.shape[0] / 2 * 1.2)
          d = int(out_i.shape[1] / 2 * 1.2)
          cv2.imwrite(f"{save_name_temp}_2.png", out_i)
          if not np.mean(out_i[a:c, b:d, 2]) == 255:
            out_i_2 = self.routine.rgbscale(self.routine.getCluster(image, idx, 2, (0, 1, 0)))
            cv2.imwrite(f"{save_name_temp}_2_t.png", out_i_2)
            cluster = 2
          
          
          
          ind, rm = self.routine.sum_idx_array(self.routine.realIdx(idx, int(cluster)))
          bacteria.sample_cluster = self.routine.rev_idx_array(ind, rm)
          print(bacteria.sample_cluster)
          

          out_i = self.routine.getCluster(image, bacteria.sample_cluster, 1, (1, 0, 0))

          plt.imshow(self.routine.rgbscale(out_i))
        #   plt.show()
          save_name_temp=SAVE_NAME+'-2-'+sample
        #   image = self.visualize_images(self.folder, self.samples, bacteria.normalized, bacteria.sample_cluster, save_name_temp)
          out_i_visua = self.routine.getCluster(bacteria.normalized[50,:,:], bacteria.sample_cluster, 0, (1, 1, 1))
      
          cv2.imwrite(f"{save_name_temp}.png", self.routine.rgbscale(out_i_visua))

          bacteria.image = None
          bacteria.save()
            



    def results(self,
                work_dir: str,
                case=0,
                spectral_range=(1, 241),
                true_labels=None):

        out_dir = 'outputs'
        models = load(os.path.join(out_dir, 'models_pipeline_01.joblib'))
        with open(os.path.join(os.getcwd(), out_dir, 'config.json'), 'r') as f_cfg:
            cfg = json.load(f_cfg)
            f_cfg.close()

        for sample in list(self.samples.keys()):
            for idx, model in enumerate(models):
                print(model.__class__.__name__, sample)

                bacteria = Utils.load_hsi_sample(path=self.folder, name=sample)
                matrix = self._signal_filter(sample=bacteria)
                matrix = matrix[:, spectral_range[0]:spectral_range[1]]

                ind, rem = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))

                result = model.predict(matrix[ind])
                full_array = self.routine.rev_idx_array(ind, rem, tfill=result)

                targets = []
                cl_legends = []
                image = self.routine.matrix2hsi(matrix, *bacteria.normalized.shape[1:])[50, :, :]
                image = self.routine.getCluster(image, bacteria.sample_cluster, 0, (255, 255, 255))

                for classe in model.classes_:
                    #             print(hex2rgb(colors[str(int(classe))]), colors[str(int(classe))])
                    image = self.routine.getCluster(image,
                                                    full_array,
                                                    classe,
                                                    Utils.hex2rgb(self.utils.colors[str(int(classe))]))
                    cl_legends.append(self.utils.colors[str(int(classe))])

                    if not true_labels:
                        targets.append(Utils.get_name(cfg['samples_training'], classe, 0))
                    else:
                        targets.append(true_labels[classe])

                cl_legends = [self.utils.colors[str(int(classe))] for classe in model.classes_]
                patches = [mpatches.Patch(color=cl_legends[i], label=targets[i])
                           for i in range(len(targets))]

                fig, ax = plt.subplots(**{'figsize': (14, 8), 'dpi': 300})

                ax.axis('off')
                ax.imshow(np.uint8(image))
                legend = fig.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                # os.makedirs('{}/sample/{}'.format(fig_dest_dir, model.__class__.__name__ + '_' + str(idx)),
                #             exist_ok=True)
                # fig.savefig('{}/sample/{}/{}.jpg'
                #             .format(fig_dest_dir, model.__class__.__name__ + '_' + str(idx), sample),
                #             bbox_extra_artists=[legend], bbox_inches='tight')
                # fig.savefig('{}/{}.jpg'
                #             .format(work_dir, '-7-' + model.__class__.__name__ + '_' + str(idx) + sample),
                #             bbox_extra_artists=[legend], bbox_inches='tight')
               
                fig.savefig(
                    # os.path.join(out_dir, '_LEGEND_' + file_name),
                    work_dir+'-7-' + model.__class__.__name__ + '_' + str(idx) + '-' + sample,
                    bbox_inches='tight',
                    bbox_extra_artists=[legend],
                )

                # plt.show()

                del ind, bacteria, matrix

    def get_Xy(self, case: int, spectral_range=(1, 241), test_size=0.5, true_labels=None):


        y_test = np.array([])
        y_train = np.array([])

        X_test = np.array([]).reshape(0, spectral_range[1] - spectral_range[0])
        X_train = np.array([]).reshape(0, spectral_range[1] - spectral_range[0])

        target_names = []
        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_hsi_sample(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]

            if case != 0:
                if true_labels:
                    for key in list(true_labels.keys()):
                        if true_labels[key] == self.samples[sample][case]:
                            target_names.append((sample, key))
                else:
                    target_names.append(Utils.get_name(self.samples,
                                                       self.samples[sample][case], case))
            else:
                target_names.append(sample)

            ind, _ = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))

            if test_size < 1.0:
                idx_train, idx_test = train_test_split(ind, test_size=test_size, shuffle=False)
                X_test = np.concatenate((X_test, matrix[idx_test]))
                X_train = np.concatenate((X_train, matrix[idx_train]))

                y = np.ones(idx_test.shape) * self.samples[sample][case]
                y_test = np.concatenate((y_test, y))

                y = np.ones(idx_train.shape) * self.samples[sample][case]
                y_train = np.concatenate((y_train, y))

            if test_size == 1.0:
                X_test = np.concatenate((X_test, matrix[ind]))
                y_test = np.concatenate((y_test, np.ones(ind.shape) * self.samples[sample][case]))

        if test_size < 1.0:
            X_train, y_train = shuffle(X_train, y_train)
            return X_train, X_test, y_train, y_test, target_names

        X_test, y_test = shuffle(X_test, y_test)

        return X_test, y_test, target_names

    @staticmethod
    def train_models(file_name, x_train: np.ndarray, x_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     models: list, target_names: list, models_file: str,
                     samples_dict: dict, work_dir='outputs'):


        os.makedirs(os.path.join(os.getcwd(), work_dir), exist_ok=True)
        result= {}
        config = dict({'samples_training': samples_dict})
        for key in config.keys():
            for sample in config[key].keys():
                config[key][sample] = [int(case) for case in config[key][sample]]

        with open(os.path.join(os.getcwd(), work_dir, 'config.json'), 'w') as f_cfg:
            json.dump(config, f_cfg)
            f_cfg.close()

        new_target = []
        for key in target_names:
            if type(key) is tuple:
                if key[1] not in new_target:
                    new_target.append(key[1])

        for model in models:
            model_name = model.__class__.__name__
            print(model_name)
            # print(model.__class__.__name__)

            classifier = model.fit(x_train, y_train)
            predictions = classifier.predict(x_test)

            classification_score = classification_report(y_test,
                                        predictions,
                                        target_names=list(new_target if len(new_target) > 0 else target_names), output_dict=True)
            print(classification_score)

            # print(classification_report(y_test,
            #                             predictions,
            #                             target_names=list(new_target if len(new_target) > 0 else target_names)))

            _, ax = plt.subplots(figsize=(13, 10))
            sns.set(font_scale=1.5)
            cf_matrix = confusion_matrix(y_test, predictions)
            sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                        xticklabels=new_target if len(new_target) > 0 else target_names,
                        yticklabels=new_target if len(new_target) > 0 else target_names,
                        fmt='.2%', cmap='Blues', ax=ax, annot_kws={"size": 16})
            
            result[model_name] = classification_score
            plt.savefig(file_name+model_name+'_confusion_matrix.jpeg', dpi=500)
          
        dump(models, os.path.join(os.getcwd(), work_dir, models_file))
        
        return result


    def retornar_espectro(self, SAVE_NAME):
        save_name_temp=SAVE_NAME+'-3-'+'mean_spectres.jpeg'

        true_labels = {'positiva': 1, 'negativa': 2}
        mean_matrix, targets = self.get_group_mean_matrix(case=0, true_labels=true_labels)
        self.plot_spectres(mean_group=mean_matrix,
                       true_labels=targets,
                       out_dir=SAVE_NAME,
                       file_name=save_name_temp)
        
    def get_training(self, samples_dict, SAVE_NAME):
        
        X_train, X_test, y_train, y_test, target_names = self.get_Xy(case=0, spectral_range=(1, 241))
       
        models_file = 'models_pipeline_01.joblib'
        seed = 42
        models =[
            discriminant_analysis.LinearDiscriminantAnalysis(covariance_estimator=None, 
                                                            n_components=None,
                                                            priors=None, shrinkage=None, 
                                                            solver='svd',store_covariance=False,
                                                            tol=0.0001),
            discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, 
                                                                store_covariance=False, 
                                                                tol=0.0001),
            ]

        for model in models:
            print(model.__class__.__name__)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)


        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test,y_pred)
        save_name_temp=SAVE_NAME+'-5-'
        out_dir = 'outputs'
        results = self.train_models(x_train=X_train,
                            x_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            models=models,
                            samples_dict=samples_dict,
                            target_names=target_names,
                            models_file=models_file,
                            work_dir=out_dir,file_name=save_name_temp)
        return results
    
    def get_pca(self, SAVE_NAME):
        save_name_temp=SAVE_NAME+'-6-'+'pca_grath.jpeg'
        true_labels = {'positiva': 1, 'negativa': 2}
        pca_matrix, var_exp, target_groups = self.get_pca_matrix(case=0, true_labels=true_labels, 
                                                             mean=True)
        self.plot_pca_samples(pca_matrix_groups=pca_matrix, 
                          exp_var_groups=var_exp, 
                          true_labels=target_groups, 
                          out_dir=SAVE_NAME, file_name=save_name_temp)

    # TODO: classification report file
