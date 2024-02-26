use plotters::prelude::*;


pub fn plot_loss(train_loss: Vec<f32>, val_loss: Vec<(f32, f32)>) -> Result<(), Box<dyn std::error::Error>> {
    let length: f32 = train_loss.len() as f32;
    let height: f32 = 3.0;
    let filename = "loss.png";

    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..length, -0f32..height)?;

    chart.configure_mesh().draw()?;

    let train_loss_series = LineSeries::new(train_loss.iter().enumerate().map(|(i, x)| (i as f32, *x)), &BLACK);
    let val_loss_series = LineSeries::new(val_loss, &RED);

    chart
        .draw_series(train_loss_series)?
        .label("Training loss")
        .legend(|(x,y)| PathElement::new(vec![(x,y), (x+20, y)], &BLACK));

    chart
        .draw_series(val_loss_series)?
        .label("Validation loss")
        .legend(|(x,y)| PathElement::new(vec![(x,y), (x+20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Plot saved to {}", filename);

    Ok(())
}
