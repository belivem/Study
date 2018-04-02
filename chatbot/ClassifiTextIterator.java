package com.huawei.algorithm.lab.dlp4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;


public class ClassifiTextIterator implements DataSetIterator{
		

	private static final long serialVersionUID = 1L;

	private Map<String,Integer> class2Index = null;					//class convert to index
	private Map<String,Integer> validateWord = null;				//define the validate pinyin
	private List<String> data = null;								//Data, <Intent,List<Content>>
	private List<String> labels;
	
	private final String separator;
	private final int vectorSize;
	private final int labelCount;
	private final int batchSize;
	private final int truncateLength;
	private int cursor = 0;
	
	public int getVectorSize(){
		return this.vectorSize;
	}
	
	public int getLabelCount(){
		return this.labelCount;
	}
	
	public int getTruncateLength(){
		return this.truncateLength;
	}
	
	public ClassifiTextIterator(Map<String,Integer> validateWord, Map<String,ArrayList<String>> data, int batch, int truncateLength,String separator){
		this.validateWord = validateWord;
		this.data = flatMapDate(data);
		
		List<String> labels = new ArrayList<String>();
		this.class2Index = generateClass2Index(data,labels);
		this.labels = labels;
		
		this.labelCount = this.class2Index.size();
		this.separator = separator;
		this.vectorSize = validateWord.size();
		this.batchSize = batch;
		this.truncateLength = truncateLength;
	}
	
	public List<String> flatMapDate(Map<String,ArrayList<String>> data){
		List<String> list = new ArrayList<String>();		
		
		for(Map.Entry<String, ArrayList<String>> entry:data.entrySet()){
			String intent = entry.getKey();
			List<String> contents = entry.getValue();
			
			for(String line : contents){
				String newLine = line +"\t"+intent;
				list.add(newLine);
			}			
		}
		
		Collections.shuffle(list);
		
		return list;
	}
	
	@Override
	public DataSet next(int batch) {
		// TODO Auto-generated method stub
		
		if(cursor >= totalExamples()){
			System.out.println("Cursor is out of range ! Exceed the total examples...");
			return null;
		}
		
		return nextDataSet(batch);		
	}
	
	 private DataSet nextDataSet(int num){
		 
		 int totalExamsple = totalExamples();
		 int lastNum = Math.min(num,totalExamsple - cursor);
		 
		 List<String> contents = new ArrayList<>(lastNum);
		 int[] labelIndex = new int[lastNum];		 
		 
		 int maxLen = 0;
		 for(int i = 0; i < lastNum;i++){
			 String line = this.data.get(this.cursor);
			 String content = line.split("\t")[0];
			 String intent  = line.split("\t")[1];
			 
			 //System.out.println("index ,"+i+", content ==> "+content+", intent ==> "+intent);
			 
			 maxLen = Math.max(maxLen, content.split(this.separator).length);
			 
			 contents.add(content);
			 labelIndex[i] = this.class2Index.get(intent);			 
			 ++cursor;
		 }
		 
		 if(maxLen > truncateLength)
			 maxLen = truncateLength;
		 
		 //features and labels
		 INDArray features = Nd4j.create(new int[]{lastNum, vectorSize, maxLen}, 'f');
		 INDArray labels   = Nd4j.create(new int[]{lastNum, this.labelCount, maxLen},'f');
			 
		 INDArray featuresMask = Nd4j.create(new int[]{lastNum, maxLen}, 'f');
		 INDArray labelsMask   = Nd4j.create(new int[]{lastNum, maxLen},'f');
		 
		 int[] index_mask = new int[2];
		 for(int i = 0; i < lastNum;i++){			 
			 index_mask[0] = i;			 
			 String[] words = contents.get(i).split(this.separator);			 
			 //echo word in content
			 for(int j = 0; j < words.length && j < maxLen;j++){				 
				 index_mask[1] = j;				 				 
				 int feature_index = this.validateWord.get(words[j]);
				 features.putScalar(new int[]{i,feature_index,j}, 1.0);				 
				 featuresMask.putScalar(index_mask, 1.0);
			 }
			 
			int label_index = labelIndex[i];
			int lastIdx = Math.min(words.length, maxLen);
			labels.putScalar(new int[]{i, label_index, lastIdx - 1}, 1.0);
			labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
		 }
		 
		 DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);
		 return ds;
	 }
		 

	
	public Map<String,Integer> generateClass2Index(Map<String,ArrayList<String>> data,List<String> labels){
		
		Map<String,Integer> map = new HashMap<String,Integer>();
		int count = 0;
		for(String key : data.keySet()){
			labels.add(key);
			map.put(key, count);
			++count;
		}
		
		return map;
	}
	
	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return cursor < numExamples();
	}

	@Override
	public DataSet next() {
		// TODO Auto-generated method stub
		return next(this.batchSize);
	}

	@Override
	public int totalExamples() {
		// TODO Auto-generated method stub		
		return data.size();
	}

	@Override
	public int inputColumns() {
		// TODO Auto-generated method stub
		return this.vectorSize;
	}

	@Override
	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return this.class2Index.size();
	}

	@Override
	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		cursor = 0;
		
	}

	@Override
	public int batch() {
		// TODO Auto-generated method stub
		return this.batchSize;
	}

	@Override
	public int cursor() {
		// TODO Auto-generated method stub
		return cursor;
	}

	@Override
	public int numExamples() {
		// TODO Auto-generated method stub
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return this.labels;
	}
}

