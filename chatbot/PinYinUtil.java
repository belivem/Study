package com.huawei.algorithm.lab.IOManager;

import java.util.List;

import com.hankcs.hanlp.dictionary.py.Pinyin;
import com.hankcs.hanlp.dictionary.py.PinyinDictionary;

public class PinYinUtil {

	
	public static String convert2PinyinWithoutTone(String str, String separator) {

		if (str == null || str.length() == 0) {
			return str;
		}

		StringBuilder builder = new StringBuilder();
		List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(str, true);

		int length = pinyinList.size();
		int i = 1;
		for (Pinyin pinyin : pinyinList) {
			if (pinyin == Pinyin.none5) {
				builder.append(str.charAt(i - 1));
			} else
				builder.append(pinyin.getPinyinWithoutTone());

			if (i < length) {
				builder.append(separator);
			}
			++i;
		}

		String pinyin_str = builder.toString().trim();
		return pinyin_str;
	}

	
	public static String convert2PinyinWithTone(String str, String separator) {

		if (str == null || str.length() == 0) {
			return str;
		}

		StringBuilder builder = new StringBuilder();
		List<Pinyin> pinyinList = PinyinDictionary.convertToPinyin(str, true);

		int length = pinyinList.size();
		int i = 1;
		for (Pinyin pinyin : pinyinList) {
			if (pinyin == Pinyin.none5) {
				builder.append(str.charAt(i - 1));
			} else
				builder.append(pinyin.toString());

			if (i < length) {
				builder.append(separator);
			}
			++i;
		}

		String pinyin_str = builder.toString().trim();
		return pinyin_str;

	}
	
}
