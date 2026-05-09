# Generator Overfit Check

This is a small debugging run only; it does not replace the full generator dataset.

Train examples: `96`
Validation examples: `96`
Final train loss: `2.9445869674285254`
Final eval loss: `2.5046396305163703`
Finite losses: `True`

## Sample Predictions

- Query: `What do I do if I have questions?`
  Reference: `All model year 1996 and older diesel powered vehicles registered anywhere in New York State with a gross vehicle weight rating GVWR less than 8501 pounds are not subject to the OBDII emissions inspection, the low enhanced emissions inspection or the diesel emissions inspection.`
  Prediction: `If you have questions, you can contact the OBDII emissions inspection or the diesel emissions inspection. If you have questions, you can contact the OBDII emissions inspection or the diesel emissions inspection.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `What are the stipulation of diesel powered vehicles and inspections?`
  Reference: `All model year 1996 and older diesel powered vehicles registered anywhere in New York State with a gross vehicle weight rating GVWR less than 8501 pounds are not subject to the OBDII emissions inspection, the low enhanced emissions inspection or the diesel emissions inspection.`
  Prediction: `If you have a diesel powered vehicle, you may be subject to either the OBDII emissions inspection or the diesel emissions inspection.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `What is that number to call for questions?`
  Reference: `The NY State Dept of Environmental Conservation, the NY State Dept of Transportation or the NY State Police can test any diesel powered vehicle that has a MGW of more that 8500 lbs.`
  Prediction: `If you have a diesel powered vehicle, you can call the OBDII emissions inspection or the diesel emissions inspection.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `Where can I get my vehicle tested?`
  Reference: `The NY State Dept of Environmental Conservation, the NY State Dept of Transportation or the NY State Police can test any diesel powered vehicle that has a MGW of more that 8500 lbs.`
  Prediction: `You can get your vehicle tested at a gas station or a gas station.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `Please tell me about the diesel opacity emissions inspection.`
  Reference: `this inspection is not required in other areas of NY states or for vehicles registered outside of the state. Nonetheless, if the vehicle has an MGW of more than 8500 lbs then the NY State Department of Environmental Conservation, the NY State Department of Transportation, or the NY State Police can do a test for any diesel-powered vehicle.`
  Prediction: `If you have a diesel powered vehicle, you may be subject to the OBDII emissions inspection or the diesel emissions inspection. If you have a diesel powered vehicle, you may be subject to the OBDII emissions inspection or the diesel emissions inspection.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `No, it didn't fail any inspection.`
  Reference: `The diesel emissions inspection is made at the annual safety inspection and it's done at a diesel emissions inspection station.`
  Prediction: `Based on the support article, a non - exempt diesel - powered motor vehicle may be subject to either the OBDII emissions inspection or the diesel emissions inspection.`
  Status/model: `ok` / `extractive_synthesizer`

- Query: `Can you tell me when and where a diesel inspection can happen?`
  Reference: `The diesel emissions inspection is made at the annual safety inspection and it's done at a diesel emissions inspection station.`
  Prediction: `The diesel emissions inspection can take place at any time of the day or night.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `I have another question for about Investigation process. What happens after the investigation is completed?`
  Reference: `This will depend on what is found in the investigation. The result could be no action at all, a warning letter Notice of Investigation put in your file, or a formal administrative hearing before an Administrative Law Judge.`
  Prediction: `If you have a complaint against an inspection station or certified inspector, you can file a complaint against the inspection station or certified inspector.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `They said something about a hearing waiver where I would just pay a monetary penalty. What if I don't accept the waiver?`
  Reference: `If you reject their offer, a hearing must be held.`
  Prediction: `If you don't accept the waiver, you can file a complaint against a DMV regulated automotive business.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`

- Query: `What does the hearing process entail?`
  Reference: `During the hearing process, you are entitled to hire an attorney and to have your attorney present at the hearing. The DMV will be the petitioner in this case and your complainant may be called as a witness.`
  Prediction: `The hearing process is entails a complaint against an inspection station or certified inspector.`
  Status/model: `ok` / `outputs/generator/flan_t5_overfit_check`
