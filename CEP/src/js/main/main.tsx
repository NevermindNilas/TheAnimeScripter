import { useState } from "react";
import { TextField, Flex, ActionButton, Picker, Item,
         Selection, darkTheme, Provider, Checkbox, CheckboxGroup, TooltipTrigger, Tooltip, Slider } from "@adobe/react-spectrum";
//import icon from adobe spectrum
import FolderAdd from "@spectrum-icons/workflow/FolderAdd";
import Play from "@spectrum-icons/workflow/Play";
import PyInterface from "./PyInterface";

//import the host stuf from ad
const Main = () => {
  const [outputFolder, setOutputFolder] = useState("C:/Users/YourName/Desktop/Output");
  const UpScaleOptions = [
    {id: 1, name: "ShuffleCugan"},
    {id: 2, name: "Compact"},
    {id: 3, name: "UltraCompact"},
    {id: 4, name: "SuperUltraCompact"},
    {id: 5, name: "Cugan"},
    {id: 6, name: "Cugan-amd"}
  ];
  const CuganDenoiseOptions = [
    {id: 1, name: "No Denoise"},
    {id: 2, name: "Conservative"},
    {id: 3, name: "Denoise 1x"},
    {id: 4, name: "Denoise 2x"},
  ];
  const [denoiseOption, setDenoiseOption] = useState(CuganDenoiseOptions[0]);
  const [upScaleModel, setUpScaleModel] = useState(UpScaleOptions[0]);
  const [selected, setSelected] = useState([]);
  const [upscaleValue, setUpscaleValue] = useState(2);
  const [interpolateValue, setInterpolateValue] = useState(2);

  const handleSelectionChange = (item: any) => {
    setUpScaleModel(item);
    upScaleModel.name = item.name;
  }

  const handleDenoiseChange = (item: any) => {
    setDenoiseOption(item);
    denoiseOption.name = item.name;
  }

  const selectOutputFolder = () => {
    const folder = window.cep.fs.showOpenDialog(false, true, "Select Output Folder", null, null);
    if (folder.err === 0) {
      setOutputFolder(folder.data);
    }
  }
  
  const start = async () => {
    var pyi = new PyInterface('TheAnimeScripter');
    await pyi.connect();

  }

  const test = async () => {
    const data = {
      outputFolder: outputFolder,
      Deduplicatevalue: selected.includes("1" as never),
      Upscalevalue: selected.includes("2" as never),
      Interpolatevalue: selected.includes("3" as never),
      upscaleValue: upscaleValue,
      interpolateValue: interpolateValue,
      upScaleModel: upScaleModel.name,
      denoiseOption: denoiseOption.name
    }
    console.log(JSON.stringify(data));
  }
    
  return (
    <div>
      <Provider theme={darkTheme}>
      <Flex direction="row" alignItems = "center">
      <Flex direction="column" gap="size-100" alignItems = "center">
        <Flex direction="row" gap="size-100" alignItems = "center">
        <TextField
          aria-label="Output Folder"
          defaultValue="C:/Users/YourName/Desktop/Output"
          value={outputFolder}
          onChange={setOutputFolder}
          width = '75vw'
        />
        <TooltipTrigger delay={0}>
      <ActionButton aria-label = "Set Output Folder" width = '15vw' onPress={selectOutputFolder}>
          <FolderAdd />
        </ActionButton>
        <Tooltip>Set Output Folder</Tooltip>
      </TooltipTrigger>
        </Flex>
        <CheckboxGroup
          aria-label="Checkbox group"
          value={selected}
          onChange={setSelected}
          orientation="horizontal"
      >
        <Checkbox value="1">DeDuplicate</Checkbox>
        <Checkbox value="2">Upscale</Checkbox>
        <Checkbox value="3">Interpolate</Checkbox>
      </CheckboxGroup>
        <Picker label="UpScale Models" width = '90vw' defaultSelectedKey="ShuffleCugan" items = {UpScaleOptions} 
        onSelectionChange={handleSelectionChange}>
        {(item: { name: any; }) => <Item>{item.name}</Item>}
        </Picker>
        <Picker label="Denoise Options" width = '90vw' defaultSelectedKey="No Denoise" items = {CuganDenoiseOptions}
        onSelectionChange={handleDenoiseChange}>
        {(item: { name: any; }) => <Item>{item.name}</Item>}
        </Picker>
        <Slider width = '90vw' label="Upscale Value" value={upscaleValue} onChange={setUpscaleValue} labelValue="Upscale Value" />
        <Slider width = '90vw' label = "Interpolate Value" value={interpolateValue} onChange={setInterpolateValue} labelValue="Interpolate Value" />
        <ActionButton aria-label = "Start" width = '85vw' onPress={test}>
        <Play/>
          Start
      </ActionButton>
      </Flex>
      </Flex>
      

      </Provider>
    </div>
  );
};
export default Main;