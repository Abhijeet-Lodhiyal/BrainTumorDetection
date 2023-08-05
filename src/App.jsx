import * as tf from '@tensorflow/tfjs';
import { useEffect, useRef, useState } from 'react';

function App() {
  const [model,setModel] = useState(null);
  const imageRef = useRef(null);
  const inputRef = useRef(null);
  const [output, setOutput] = useState('');

  const importModel = async () => {
    const newModel = await tf.loadLayersModel('http://localhost:5173/model.json');
    setModel(newModel)
    console.log('Model loaded successfully');
  };

  useEffect(() => {
    importModel();
  }, []);

  const imageLoaded = () => {
    imageRef.current.src = URL.createObjectURL(inputRef.current.files[0]);
  }

  const predictVal = async (e) => {
    e.preventDefault();

    if (!model) {
      console.error('Model not loaded yet.');
      return;
    }
    await imageRef.current.decode()
    imageRef.current.width = 200;                           // 200x200 image for model
    imageRef.current.height = 200;
    const imag = tf.browser.fromPixels(imageRef.current);  // image to tensor
    const inputWithBatch = imag.expandDims(0);             // to give batch size to tensor
    const result = await model.predict(inputWithBatch);
    const ans = result.arraySync()[0][0]
    console.log(result.arraySync()[0][0]);
    if (ans >= 0.5)
      setOutput("The model predicted that the given image has Tumor with a probablity of " + ans)
    else
      setOutput("The model predicted that the given image does not have Tumor with a probablity of " + ans )
  };
  

  return (
    <div className='flex flex-col items-center w-screen h-screen text-center bg-[#fff8ff]'>
      <h1 className='lg:text-7xl md:text-5xl text-3xl m-5 font-Bungee'>Brain Tumor detection</h1>
      <div className={`flex md:flex-row flex-col gap-5 md:h-[70%]  md:w-[70%] items-center w-full ${output === "" ? 'justify-center':''}`}>
        <div className='flex flex-col border-2 p-5 rounded-2xl md:w-[50%] w-[80%] h-full justify-center items-center gap-5 shadow-lg shadow-[#3f3f3f] border-[#3f3f3f]'>
          <form onSubmit={predictVal}>
            <label htmlFor='imageFile' className='font-Nunito border-2 pt-1 pb-2 pl-1 pr-1 border-black rounded-md m-1 hover:scale-125 hover:cursor-pointer hover:bg-black hover:text-white' >Upload</label>
            <input type='file' accept='.jpg, .png, .webp' id='imageFile' className='hidden' onChange={imageLoaded} ref={inputRef} />
            <button type='submit' className='font-Nunito border-2 p-1 border-black rounded-md m-1 hover:bg-black hover:text-white' disabled={model === null }>Submit</button>
          </form>
          <img ref={imageRef} src='https://t1.gstatic.com/licensed-image?q=tbn:ANd9GcRM0OQsITDDUQ-PCjobiXAyUfEQn1sOAkjorPKB2miR-sYx_aCjqMSevH2Y4WjIvPoA' className='w-[200px] h-[200px]' />
        </div>
        <div className={`flex flex-col border-2 p-5 rounded-2xl md:w-[50%] w-[80%] h-full justify-around items-center shadow-[#3f3f3f] border-[#3f3f3f] ${output === "" ? 'hidden':''} shadow-lg`}>
          <h1 className='lg:text-5xl md:text-3xl text-xl m-5 font-Bungee'>Predictions</h1>
          <h1 className='font-Nunito font-medium md:text-3xl text-xl'>{output}</h1>
          <p></p>
        </div>
      </div>


    </div>
  );
}

export default App;
