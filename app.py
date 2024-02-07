import gradio as gr
import os
import datetime
import pytz
from pathlib import Path

def current_time():
	current = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y年-%m月-%d日 %H时:%M分:%S秒")
	return current

print(f"[{current_time()}] 开始部署空间...")

"""
print(f"[{current_time()}] 日志：安装 - 必要包")
os.system("pip install -r ./requirements.txt")
"""
print(f"[{current_time()}] 日志：安装 - gsutil")
os.system("pip install gsutil")
print(f"[{current_time()}] 日志：Git - 克隆 Github 的 T5X 训练框架到当前目录")
os.system("git clone --branch=main https://github.com/google-research/t5x")
print(f"[{current_time()}] 日志：文件 - 移动 t5x 到当前目录并重命名为 t5x_tmp 并删除")
os.system("mv t5x t5x_tmp; mv t5x_tmp/* .; rm -r t5x_tmp")
print(f"[{current_time()}] 日志：编辑 - 替换 setup.py 内的文本“jax[tpu]”为“jax”")
os.system("sed -i 's:jax\[tpu\]:jax:' setup.py")
print(f"[{current_time()}] 日志：Python - 使用 pip 安装 当前目录内的 Python 包")
os.system("python3 -m pip install -e .")
print(f"[{current_time()}] 日志：Python - 更新 Python 包管理器 pip")
os.system("python3 -m pip install --upgrade pip")
print(f"[{current_time()}] 日志：安装 - langchain")
os.system("pip install langchain")
print(f"[{current_time()}] 日志：安装 - sentence-transformers")
os.system("pip install sentence-transformers")

# 安装 airio
print(f"[{current_time()}] 日志：Git - 克隆 Github 的 airio 到当前目录")
os.system("git clone --branch=main https://github.com/google/airio")
print(f"[{current_time()}] 日志：文件 - 移动 airio 到当前目录并重命名为 airio_tmp 并删除")
os.system("mv airio airio_tmp; mv airio_tmp/* .; rm -r airio_tmp")
print(f"[{current_time()}] 日志：Python - 使用 pip 安装 当前目录内的 Python 包")
os.system("python3 -m pip install -e .")

# 安装 mt3
print(f"[{current_time()}] 日志：Git - 克隆 Github 的 MT3 模型到当前目录")
os.system("git clone --branch=main https://github.com/magenta/mt3")
print(f"[{current_time()}] 日志：文件 - 移动 mt3 到当前目录并重命名为 mt3_tmp 并删除")
os.system("mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp")
print(f"[{current_time()}] 日志：Python - 使用 pip 从 storage.googleapis.com 安装 jax[cuda11_local] nest-asyncio pyfluidsynth")
os.system("python3 -m pip install jax[cuda11_local] nest-asyncio pyfluidsynth==1.3.0 -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
print(f"[{current_time()}] 日志：Python - 使用 pip 安装 当前目录内的 Python 包")
os.system("python3 -m pip install -e .")
print(f"[{current_time()}] 日志：安装 - TensorFlow CPU")
os.system("pip install tensorflow_cpu")

# 复制检查点
print(f"[{current_time()}] 日志：gsutil - 复制 MT3 检查点到当前目录")
os.system("gsutil -q -m cp -r gs://mt3/checkpoints .")

# 复制 soundfont 文件（原始文件来自 https://sites.google.com/site/soundfonts4u）
print(f"[{current_time()}] 日志：gsutil - 复制 SoundFont 文件到当前目录")
os.system("gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .")

#@title 导入和定义
print(f"[{current_time()}] 日志：导入 - 必要工具")
import functools
import os
import numpy as np
import tensorflow.compat.v2 as tf

import functools
import gin
import jax
import librosa
import note_seq

import seqio
import t5
import t5x

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

import nest_asyncio
nest_asyncio.apply()

SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'

def upload_audio(audio, sample_rate):
	return note_seq.audio_io.wav_data_to_samples_librosa(
		audio, sample_rate=sample_rate)


print(f"[{current_time()}] 日志：开始包装模型...")
class InferenceModel(object):
	"""音乐转录的 T5X 模型包装器。"""

	def __init__(self, checkpoint_path, model_type='mt3'):

		# 模型常量。
		if model_type == 'ismir2021':
			num_velocity_bins = 127
			self.encoding_spec = note_sequences.NoteEncodingSpec
			self.inputs_length = 512
		elif model_type == 'mt3':
			num_velocity_bins = 1
			self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
			self.inputs_length = 256
		else:
			raise ValueError('unknown model_type: %s' % model_type)

		gin_files = ['/home/user/app/mt3/gin/model.gin',
					'/home/user/app/mt3/gin/mt3.gin']

		self.batch_size = 8
		self.outputs_length = 1024
		self.sequence_length = {'inputs': self.inputs_length,
								'targets': self.outputs_length}

		self.partitioner = t5x.partitioning.PjitPartitioner(
				model_parallel_submesh=None, num_partitions=1)

		# 构建编解码器和词汇表。
		print(f"[{current_time()}] 日志：构建编解码器")
		self.spectrogram_config = spectrograms.SpectrogramConfig()
		self.codec = vocabularies.build_codec(
				vocab_config=vocabularies.VocabularyConfig(
				num_velocity_bins=num_velocity_bins)
				)
		self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
		self.output_features = {
				'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
				'targets': seqio.Feature(vocabulary=self.vocabulary),
		}

		# 创建 T5X 模型。
		print(f"[{current_time()}] 日志：创建 T5X 模型")
		self._parse_gin(gin_files)
		self.model = self._load_model()

		# 从检查点中恢复。
		print(f"[{current_time()}] 日志：恢复模型检查点")
		self.restore_from_checkpoint(checkpoint_path)

	@property
	def input_shapes(self):
		return {
					'encoder_input_tokens': (self.batch_size, self.inputs_length),
					'decoder_input_tokens': (self.batch_size, self.outputs_length)
		}

	def _parse_gin(self, gin_files):
		"""解析用于训练模型的 gin 文件。"""
		print(f"[{current_time()}] 日志：解析 gin 文件")
		gin_bindings = [
				'from __gin__ import dynamic_registration',
				'from mt3 import vocabularies',
				'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
				'vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'
		]
		with gin.unlock_config():
			gin.parse_config_files_and_bindings(
					gin_files, gin_bindings, finalize_config=False)

	def _load_model(self):
		"""在解析训练 gin 配置后加载 T5X `Model`。"""
		print(f"[{current_time()}] 日志：加载 T5X 模型")
		model_config = gin.get_configurable(network.T5Config)()
		module = network.Transformer(config=model_config)
		return models.ContinuousInputsEncoderDecoderModel(
				module=module,
				input_vocabulary=self.output_features['inputs'].vocabulary,
				output_vocabulary=self.output_features['targets'].vocabulary,
				optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
				input_depth=spectrograms.input_depth(self.spectrogram_config))


	def restore_from_checkpoint(self, checkpoint_path):
		"""从检查点中恢复训练状态，重置 self._predict_fn()。"""
		print(f"[{current_time()}] 日志：从检查点恢复训练状态")
		train_state_initializer = t5x.utils.TrainStateInitializer(
			optimizer_def=self.model.optimizer_def,
			init_fn=self.model.get_initial_variables,
			input_shapes=self.input_shapes,
			partitioner=self.partitioner)

		restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
				path=checkpoint_path, mode='specific', dtype='float32')

		train_state_axes = train_state_initializer.train_state_axes
		self._predict_fn = self._get_predict_fn(train_state_axes)
		self._train_state = train_state_initializer.from_checkpoint_or_scratch(
				[restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

	@functools.lru_cache()
	def _get_predict_fn(self, train_state_axes):
		"""生成一个分区的预测函数用于解码。"""
		print(f"[{current_time()}] 日志：生成用于解码的预测函数")
		def partial_predict_fn(params, batch, decode_rng):
			return self.model.predict_batch_with_aux(
					params, batch, decoder_params={'decode_rng': None})
		return self.partitioner.partition(
				partial_predict_fn,
				in_axis_resources=(
						train_state_axes.params,
						t5x.partitioning.PartitionSpec('data',), None),
				out_axis_resources=t5x.partitioning.PartitionSpec('data',)
		)

	def predict_tokens(self, batch, seed=0):
		"""从预处理的数据集批次中预测 tokens。"""
		print(f"[{current_time()}] 运行：从预处理数据集中预测音符序列")
		prediction, _ = self._predict_fn(
self._train_state.params, batch, jax.random.PRNGKey(seed))
		return self.vocabulary.decode_tf(prediction).numpy()

	def __call__(self, audio):
		"""从音频样本推断出音符序列。

		参数：
			audio：16kHz 的单个音频样本的 1 维 numpy 数组。
		返回：
			转录音频的音符序列。
		"""
		print(f"[{current_time()}] 运行：从音频样本中推断音符序列")
		ds = self.audio_to_dataset(audio)
		ds = self.preprocess(ds)

		model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
				ds, task_feature_lengths=self.sequence_length)
		model_ds = model_ds.batch(self.batch_size)

		inferences = (tokens for batch in model_ds.as_numpy_iterator()
									for tokens in self.predict_tokens(batch))

		predictions = []
		for example, tokens in zip(ds.as_numpy_iterator(), inferences):
			predictions.append(self.postprocess(tokens, example))

		result = metrics_utils.event_predictions_to_ns(
				predictions, codec=self.codec, encoding_spec=self.encoding_spec)
		return result['est_ns']

	def audio_to_dataset(self, audio):
		"""从输入音频创建一个包含频谱图的 TF Dataset。"""
		print(f"[{current_time()}] 运行：从音频创建包含频谱图的 TF Dataset")
		frames, frame_times = self._audio_to_frames(audio)
		return tf.data.Dataset.from_tensors({
				'inputs': frames,
				'input_times': frame_times,
		})

	def _audio_to_frames(self, audio):
		"""从音频计算频谱图帧。"""
		print(f"[{current_time()}] 运行：从音频计算频谱图帧")
		frame_size = self.spectrogram_config.hop_width
		padding = [0, frame_size - len(audio) % frame_size]
		audio = np.pad(audio, padding, mode='constant')
		frames = spectrograms.split_audio(audio, self.spectrogram_config)
		num_frames = len(audio) // frame_size
		times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
		return frames, times

	def preprocess(self, ds):
		pp_chain = [
				functools.partial(
						t5.data.preprocessors.split_tokens_to_inputs_length,
						sequence_length=self.sequence_length,
						output_features=self.output_features,
						feature_key='inputs',
						additional_feature_keys=['input_times']),
				# 在训练期间进行缓存。
				preprocessors.add_dummy_targets,
				functools.partial(
						preprocessors.compute_spectrograms,
						spectrogram_config=self.spectrogram_config)
		]
		for pp in pp_chain:
			ds = pp(ds)
		return ds

	def postprocess(self, tokens, example):
		tokens = self._trim_eos(tokens)
		start_time = example['input_times'][0]
		# 向下取整到最接近的符号化时间步。
		start_time -= start_time % (1 / self.codec.steps_per_second)
		return {
				'est_tokens': tokens,
				'start_time': start_time,
				# 内部 MT3 代码期望原始输入，这里不使用。
				'raw_inputs': []
		}

	@staticmethod
	def _trim_eos(tokens):
		tokens = np.array(tokens, np.int32)
		if vocabularies.DECODED_EOS_ID in tokens:
			tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
		return tokens


inference_model = InferenceModel('/home/user/app/checkpoints/mt3/', 'mt3')


def inference(audio):
	filename = os.path.basename(audio)  # 获取输入文件的文件名
	print(f"[{current_time()}] 运行：输入文件: {filename}")
	with open(audio, 'rb') as fd:
		contents = fd.read()
	audio = upload_audio(contents,sample_rate=16000)
	est_ns = inference_model(audio)
	note_seq.sequence_proto_to_midi_file(est_ns, './transcribed.mid')
	return './transcribed.mid'

title = "MT3"
description = "MT3：多任务多音轨音乐转录的 Gradio 演示。要使用它，只需上传音频文件，或点击示例以查看效果。更多信息请参阅下面的链接。"

article = "<p style='text-align: center'>出错了？试试把文件转换为MP3后再上传吧~</p><p style='text-align: center'><a href='https://arxiv.org/abs/2111.03017' target='_blank'>MT3: 多任务多音轨音乐转录</a> | <a href='https://github.com/hmjz100/mt3' target='_blank'>Github 仓库</a></p>"

examples=[['canon.flac'], ['download.wav']]

gr.Interface(
	inference,
	gr.Audio(type="filepath", label="输入"),
	outputs = gr.File(label="输出"),
	title=title,
	description=description,
	article=article,
	examples=examples
	).launch()