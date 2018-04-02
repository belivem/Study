package com.huawei.algorithm.lab.IOManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

//import com.huawei.algorithmlab.chatbot.intent.pipeline.dataset.JsonDatasetLoader;
//import com.huawei.algorithmlab.chatbot.intent.pipeline.dataset.RawDataSet;

public class InputUtil {
			
	private boolean withTone  = false;
	private String  separator = " ";
	private Map<String,Integer> validateWord = new HashMap<String,Integer>();
	
	private int validateCount = 0;
	private int truncateLength = 0;
	
	public InputUtil(int truncateLength,boolean withTone,String separator){
		this.truncateLength = truncateLength;
		this.withTone  = withTone;
		this.separator = separator;
	}
	
//	public Map<String,ArrayList<String>> getPinYinData(String input,boolean withTone,String separator){
//		Map<String,ArrayList<String>> map = new HashMap<String,ArrayList<String>>();
//		
//		RawDataSet<String, String> dataSet = JsonDatasetLoader.load(input);
//		
//		for(int i = 0;i < dataSet.getNumSamples();i++){
//			String pinyin = convert2Pinyin(dataSet.getFeature(i),withTone,separator);			
//			String intent = dataSet.getLabel(i);
//			
//			if(!map.containsKey(intent)){
//				ArrayList<String> content_list = new ArrayList<String>();
//				map.put(intent, content_list);
//			}
//			
//			map.get(intent).add(pinyin);
//			generateValidateWord(pinyin,separator);
//		}
//		
//		
//		return map;
//	}
	
	public Map<String,ArrayList<String>> getPinYinData(String inputPath,boolean withTone,String separator){
		Map<String,ArrayList<String>> map = new HashMap<String,ArrayList<String>>();
		
		File file = new File(inputPath);
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String line = null;
		try {
			line = reader.readLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		while(line != null){
			
			String[] words = line.split(",");
			if(words.length != 2){
				System.out.println("Input format is not correct!");
			}
			
			String content = words[0].trim();
			String intent  = words[1].trim();			
			String content_pinyin = convert2Pinyin(content,withTone,separator);
			
			if(!map.containsKey(intent)){
				map.put(intent, new ArrayList<String>());
			}
			
			map.get(intent).add(content_pinyin);
			generateValidateWord(content_pinyin,separator);
			try {
				line = reader.readLine();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		try {
			reader.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return map;
	}
	
	public void generateValidateWord(String pinyin,String separator){
		if(pinyin == null || pinyin.length() == 0){
			return;
		}
		
		String[] words = pinyin.split(separator);
		for(String word:words){
			if(!this.validateWord.containsKey(word)){
				this.validateWord.put(word, this.validateCount);
				++this.validateCount;
			}
		}
	}
	

	public String convert2Pinyin(String input,boolean withTone,String separator){
		if(input == null || input.length() == 0)
			return null;
		
		String pinyin = "";
		if(withTone){
			pinyin = PinYinUtil.convert2PinyinWithTone(input, separator);
		}else{
			pinyin = PinYinUtil.convert2PinyinWithoutTone(input, separator);
		}
		
		return pinyin;
	}
	
	//get total smp data: convert to pinyin already
	public Map<String,ArrayList<String>> getPinYinData(String train_path,String test_path,boolean withTone,String separator){
		
		Map<String, ArrayList<String>>	train_map = getPinYinData(train_path,withTone,separator);
		Map<String, ArrayList<String>>	test_map  = getPinYinData(test_path,withTone,separator);
	
		
		for(Map.Entry<String, ArrayList<String>> entry: test_map.entrySet()){
			String intent = entry.getKey();
			ArrayList<String> content_list = entry.getValue();
						
			for(String content:content_list){
				train_map.get(intent).add(content);
			}			
		}
		
		return train_map;
	}
	
	
	public void dataCount(Map<String,ArrayList<String>> map){
		
		int minLen = 10000;
		int maxLen = 0;
		int sumLen = 0;
		
		int count = 0;
		for(Map.Entry<String, ArrayList<String>> entry:map.entrySet()){
			
			ArrayList<String> value = entry.getValue();
			for(String line:value){
				String[] words = line.split(" ");
				minLen = Math.min(minLen, words.length);
				maxLen = Math.max(maxLen, words.length);
				sumLen += words.length;
			}
			
			count += value.size();
		}
		
		System.out.println("Key number is "+map.size()+", and value number is "+count);
		System.out.println("Line max len is ==> "+maxLen+", and min len is ==> "+minLen+", and average len is ==> "+(sumLen*1.0/count));
		
	}

	public  Map<String,Integer> getValidateWord(){
		return this.validateWord;
	}
	
	public int getValidateCount(){
		return this.validateCount;
	}
	
	public String getSeparator(){
		return this.separator;
	}
	
	public boolean getWithTone(){
		return this.withTone;
	}
	
	public int getTruncateLength(){
		return this.truncateLength;
	}
	
	public static void main(String[] args){
		
		String input_train_path = "C:\\Users\\l00369684\\Desktop\\Data\\IBM_train_dataset.csv";
		String input_test_path  = "C:\\Users\\l00369684\\Desktop\\Data\\IBM_test_dataset.csv";
		
		InputUtil util = new InputUtil(20,false," ");		
		Map<String,ArrayList<String>> total_map = util.getPinYinData(input_train_path,input_test_path,util.withTone,util.separator);
		//util.dataCount(total_map);
		
//		for(Map.Entry<String, ArrayList<String>> entry:total_map.entrySet()){
//			String intent = entry.getKey();
//			
//			for(String pinyin:entry.getValue()){
//				System.out.println(intent +" ==> "+pinyin);
//			}
//		}
		
		Map<String,Integer> validatePinyin = util.getValidateWord();
		
		System.out.println(validatePinyin.containsKey("["));
		
		System.out.println("validateCount ==> "+util.getValidateCount());
		
	}
	
}
