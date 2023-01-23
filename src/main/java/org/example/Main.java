package org.example;
/*****************************************************************************************
 * DL4J Example: version 220123
 *****************************************************************************************/
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static java.lang.System.out;
public class Main extends JComponent implements ActionListener
{
    private static int FEATURES_COUNT = 4;
    private static int CLASSES_COUNT = 3;
    private static String version = "230123";
    private static Object eval;
    private JFrame jf;
    private int x = 100;
    private int y = 100;
    private double accuracy;
    private Graphics2D g2;
    public Painter painter;
    public static void main(String[] args)
    {
        out.println("DL4J Example: version " + version);
        new Main().getGoing();
    }
    private void getGoing()
    {
        JFrame jFrame = new JFrame("WELCOME TO SUPERNN");
        jFrame.add(this);
        jFrame.setSize(1000, 1000);
        jFrame.setBackground(Color.GRAY);
        setForeground(Color.black);
        jFrame.setVisible(true);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        RegressionEvaluation eval = new RegressionEvaluation();
        BasicConfigurator.configure();
        loadData();
        out.println("\nLearningDL4J...Main thread finished normally");
    }
    private void loadData()
    {
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("iris.csv").getFile()
            ));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, 4, 3);
            DataSet allData = iterator.next();
            allData.shuffle(123);
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();
            irisNNetwork(trainingData, testingData);
        } catch (Exception e) {
            out.println("Error: " + e.getLocalizedMessage());
        }
    }
    private void irisNNetwork(DataSet trainingData, DataSet testData)
    {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.9))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        Evaluation eval = null;
        for (int i = 0; i < 1000; i++)
        {
            INDArray output = model.output(testData.getFeatureMatrix());
            eval = new Evaluation(3);
            model.fit(trainingData);
            eval.eval(testData.getLabels(), output);
            accuracy = eval.accuracy();
            out.print("\nAccuracy " + accuracy + " " + x + " " + y);
            //if (i % 5 == 0)
            {
                repaint();
            }
        }
        out.printf(eval.stats());
    }
    public void paint(Graphics g)
    {
        g.setColor(Color.black);
        g.fillOval(x, y, 10, 10);
        x += 1;
        y = (int) (accuracy * 1000);
        super.paint(g);
   }
    @Override
    public void actionPerformed(ActionEvent e)
    {
        out.println("tick");
    }
}
