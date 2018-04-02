package com.huawei.algorithm.lab.deeplearning.example;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.INDArrayIndex;

import com.huawei.algorithm.lab.IOManager.InputUtil;
import com.huawei.algorithm.lab.dlp4j.ClassifiTextIterator;

public class DataSetTest {
	
	private static final String input_train_path = "C:\\Users\\l00369684\\Desktop\\Data\\IBM_train_dataset.csv";
	private static final String input_test_path  = "C:\\Users\\l00369684\\Desktop\\Data\\IBM_test_dataset.csv";
	
	public void testUtil() throws IOException{
		int batchSize = 100;		
		InputUtil util = new InputUtil(20,false," ");
		Map<String,ArrayList<String>> dataMap = util.getPinYinData(input_train_path,input_test_path, util.getWithTone(), util.getSeparator());
		
		ClassifiTextIterator iterator = new ClassifiTextIterator(util.getValidateWord(),dataMap,batchSize,util.getTruncateLength(),util.getSeparator());
		DataSet set = iterator.next();
		List<DataSet> examplesDataSet = set.asList();
		
		System.out.println("example size    ==> "+examplesDataSet.size());
		System.out.println("Number examples ==> "+set.numExamples());
		System.out.println("Number input    ==> "+set.numInputs());
		
	}
	
	public void testIterator() throws IOException{
		int batchSize = 100;
		
		InputUtil util = new InputUtil(20,false," ");
		Map<String,ArrayList<String>> dataMap = util.getPinYinData(input_train_path,input_test_path, util.getWithTone(), util.getSeparator());
		
		ClassifiTextIterator iterator = new ClassifiTextIterator(util.getValidateWord(),dataMap,batchSize,util.getTruncateLength(),util.getSeparator());
		
		DataSet set = iterator.next(batchSize);
		INDArray features = set.getFeatures();
		INDArray labels   = set.getLabels();
		INDArray firstSentence =  features.get(new INDArrayIndex[]{point(0),all(),all()});
		INDArray firstLabel    =  labels.get(new INDArrayIndex[]{point(0),all(),all()});
		
		for(int i = 0; i < 20;i++ ){
			INDArray word = firstSentence.get(new INDArrayIndex[]{all(),point(i)}).dup();
			float[] wordData = word.data().asFloat();
			System.out.print("word ==> ");
			for(int j = 0;j < wordData.length;j++){
				System.out.print(wordData[j]+", ");
			}
			System.out.println();
		}
		
		System.out.println();
		
		for(int i = 0; i < 20;i++){
			INDArray word = firstLabel.get(new INDArrayIndex[]{all(),point(i)}).dup();
			float[] wordData = word.data().asFloat();
			System.out.print("label ==> ");
			for(int j = 0;j < wordData.length;j++){
				System.out.print(wordData[j]+", ");
			}
			System.out.println();
		}
		
		System.out.println("vector size ==> "+iterator.getVectorSize()+", label count ==> "+iterator.getLabelCount());
	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		DataSetTest classify = new DataSetTest();
		classify.testIterator();
		classify.testUtil();
	}

}
